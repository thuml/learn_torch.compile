
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


cpp_fused_convolution_backward_hardswish_backward_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(15872L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1344L*x2) + (65856L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1344L*x1)));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1344L*x2) + (65856L*x1)));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
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
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1344L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1344L*x1) + (65856L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1344L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1344L*x1) + (65856L*x0)));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp32 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp24 = static_cast<float>(0.002551020408163265);
                        auto tmp25 = at::vec::Vectorized<float>(tmp24);
                        auto tmp26 = tmp23 * tmp25;
                        auto tmp28 = tmp27 * tmp27;
                        auto tmp29 = tmp26 * tmp28;
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp31 = tmp19 - tmp30;
                        auto tmp33 = tmp32 * tmp25;
                        auto tmp34 = tmp31 - tmp33;
                        auto tmp36 = tmp27 * tmp35;
                        auto tmp37 = tmp34 * tmp36;
                        tmp37.store(out_ptr3 + static_cast<long>(x2 + (1344L*x1) + (65856L*x0)));
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
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (224L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (224L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.002551020408163265);
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1104L*x2) + (54096L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1104L*x2) + (54096L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8832L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1104L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1104L*x2) + (54096L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1104L*x2) + (54096L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1104L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1104L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1104L*x2) + (54096L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1104L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1104L*x1) + (54096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1104L*x1) + (54096L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1104L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1104L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1104L*x1) + (54096L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.002551020408163265);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (1104L*x1) + (54096L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1104L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1104L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1104L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1104L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1104L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1104L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1104L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (1104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_7 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.002551020408163265);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (184L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_10 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.002551020408163265);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_12 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (184L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.002551020408163265);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_15 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.002551020408163265);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_17 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp8 = tmp4 * tmp7;
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    tmp_acc1_vec = tmp_acc1_vec + tmp8;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (184L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (184L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp7 = tmp5 - tmp6;
                auto tmp9 = static_cast<float>(0.002551020408163265);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp7 * tmp14;
                auto tmp16 = tmp4 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                tmp22.store(out_ptr2 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_20 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.002551020408163265);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_22 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (184L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (184L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (184L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp9 = tmp7 - tmp8;
                auto tmp11 = static_cast<float>(0.002551020408163265);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp10 * tmp12;
                auto tmp15 = tmp14 * tmp14;
                auto tmp16 = tmp13 * tmp15;
                auto tmp17 = tmp9 * tmp16;
                auto tmp18 = tmp6 - tmp17;
                auto tmp20 = tmp19 * tmp12;
                auto tmp21 = tmp18 - tmp20;
                auto tmp23 = tmp14 * tmp22;
                auto tmp24 = tmp21 * tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_25 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.002551020408163265);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_27 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp8 = tmp6 + tmp7;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (184L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (184L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.002551020408163265);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (736L*x2) + (36064L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5888L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_30 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (736L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (736L*x2) + (36064L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(736L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (736L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.002551020408163265);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (736L*x1) + (36064L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (736L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (736L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(736L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.002551020408163265);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (736L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_32 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (184L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (184L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.002551020408163265);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(720L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (720L*x2) + (35280L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (720L*x2) + (35280L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5760L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_35 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(720L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (720L*x2) + (35280L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (720L*x2) + (35280L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (720L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (720L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (720L*x2) + (35280L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(720L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (720L*x1) + (35280L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (720L*x1) + (35280L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (720L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (720L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (720L*x1) + (35280L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.002551020408163265);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (720L*x1) + (35280L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(720L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(720L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(720L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (720L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (720L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (720L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(720L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(720L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (720L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (720L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (720L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0006377551020408163);
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_40 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.0006377551020408163);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_45 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.0006377551020408163);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.0006377551020408163);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp4 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_50 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.0006377551020408163);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp11 = static_cast<float>(0.0006377551020408163);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp6 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_55 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.0006377551020408163);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(188160L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0006377551020408163);
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_60 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.0006377551020408163);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (360L*x2) + (70560L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2880L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_65 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (360L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (360L*x2) + (70560L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (360L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp28 = static_cast<float>(0.0006377551020408163);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (360L*x1) + (70560L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (360L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (360L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (360L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (360L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_67 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.0006377551020408163);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (72L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (216L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (216L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_70 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (72L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.0006377551020408163);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (72L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (216L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (216L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_73 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp8 = tmp4 * tmp7;
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    tmp_acc1_vec = tmp_acc1_vec + tmp8;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (72L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (72L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp7 = tmp5 - tmp6;
                auto tmp9 = static_cast<float>(0.0006377551020408163);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp13 = tmp12 * tmp12;
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp7 * tmp14;
                auto tmp16 = tmp4 - tmp15;
                auto tmp18 = tmp17 * tmp10;
                auto tmp19 = tmp16 - tmp18;
                auto tmp21 = tmp12 * tmp20;
                auto tmp22 = tmp19 * tmp21;
                tmp22.store(out_ptr2 + static_cast<long>(x1 + (72L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (216L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (216L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_76 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    tmp_acc1_vec = tmp_acc1_vec + tmp10;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (72L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (72L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (72L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp9 = tmp7 - tmp8;
                auto tmp11 = static_cast<float>(0.0006377551020408163);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp10 * tmp12;
                auto tmp15 = tmp14 * tmp14;
                auto tmp16 = tmp13 * tmp15;
                auto tmp17 = tmp9 * tmp16;
                auto tmp18 = tmp6 - tmp17;
                auto tmp20 = tmp19 * tmp12;
                auto tmp21 = tmp18 - tmp20;
                auto tmp23 = tmp14 * tmp22;
                auto tmp24 = tmp21 * tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (216L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (216L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (216L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(216L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (216L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (216L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_79 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp8 = tmp6 + tmp7;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (72L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (72L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.0006377551020408163);
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00015943877551020407);
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_85 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(784.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(784.0);
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
                        auto tmp28 = static_cast<float>(0.00015943877551020407);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00015943877551020407);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_90 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(784.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(784.0);
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
                        auto tmp28 = static_cast<float>(0.00015943877551020407);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_92 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.00015943877551020407);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp4 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_95 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(784.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(784.0);
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
                        auto tmp28 = static_cast<float>(0.00015943877551020407);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp11 = static_cast<float>(0.00015943877551020407);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp6 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_98 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_100 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(784.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(784.0);
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
                        auto tmp28 = static_cast<float>(0.00015943877551020407);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_101 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_102 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(250880L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (40L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00015943877551020407);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_103 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_105 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(784.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(784.0);
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
                        auto tmp28 = static_cast<float>(0.00015943877551020407);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp27 * tmp29;
                        auto tmp32 = tmp31 * tmp31;
                        auto tmp33 = tmp30 * tmp32;
                        auto tmp34 = tmp26 * tmp33;
                        auto tmp35 = tmp23 - tmp34;
                        auto tmp37 = tmp36 * tmp29;
                        auto tmp38 = tmp35 - tmp37;
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_106 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_107 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(3.985969387755102e-05);
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_108 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (48L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (48L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_110 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.985969387755102e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_111 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (48L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_112 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (48L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_113 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(3.985969387755102e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp4 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_114 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (48L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_115 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (48L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp11 = static_cast<float>(3.985969387755102e-05);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp6 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_117 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_118 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(9.964923469387754e-06);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_119 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(9.964923469387754e-06);
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_120 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(9.964923469387754e-06);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_121 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(9.964923469387754e-06);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_122 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp21 = static_cast<float>(9.964923469387754e-06);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_123 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = tmp0 / tmp5;
                        auto tmp13 = static_cast<float>(0.5);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp12 + tmp14;
                        auto tmp16 = tmp11 * tmp15;
                        auto tmp17 = decltype(tmp16)::blendv(tmp11, tmp16, tmp6);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = decltype(tmp19)::blendv(tmp17, tmp19, tmp3);
                        auto tmp23 = tmp21 - tmp22;
                        auto tmp24 = tmp20 * tmp23;
                        tmp_acc0_vec = tmp_acc0_vec + tmp20;
                        tmp_acc1_vec = tmp_acc1_vec + tmp24;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp12 = tmp0 / tmp5;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 + tmp14;
                    auto tmp16 = tmp11 * tmp15;
                    auto tmp17 = decltype(tmp16)::blendv(tmp11, tmp16, tmp6);
                    auto tmp18 = static_cast<float>(0.0);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = decltype(tmp19)::blendv(tmp17, tmp19, tmp3);
                    auto tmp23 = tmp21 - tmp22;
                    auto tmp25 = static_cast<float>(9.964923469387754e-06);
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp24 * tmp26;
                    auto tmp29 = tmp28 * tmp28;
                    auto tmp30 = tmp27 * tmp29;
                    auto tmp31 = tmp23 * tmp30;
                    auto tmp32 = tmp20 - tmp31;
                    auto tmp34 = tmp33 * tmp26;
                    auto tmp35 = tmp32 - tmp34;
                    auto tmp37 = tmp28 * tmp36;
                    auto tmp38 = tmp35 * tmp37;
                    tmp38.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_264, primals_265, primals_266, primals_267, primals_269, primals_271, primals_272, primals_273, primals_274, primals_276, primals_278, primals_279, primals_280, primals_281, primals_283, primals_285, primals_286, primals_287, primals_288, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_299, primals_300, primals_301, primals_302, primals_304, primals_306, primals_307, primals_308, primals_309, primals_311, primals_313, primals_314, primals_315, primals_316, primals_318, primals_320, primals_321, primals_322, primals_323, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_334, primals_335, primals_336, primals_598, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, clone_1, div_1, convolution_2, squeeze_7, add_17, convolution_3, squeeze_10, clone_2, div_2, convolution_4, squeeze_13, add_29, convolution_5, squeeze_16, clone_3, div_3, convolution_6, squeeze_19, clone_4, div_4, convolution_7, squeeze_22, add_46, convolution_8, squeeze_25, clone_5, div_5, convolution_9, squeeze_28, clone_6, div_6, convolution_10, squeeze_31, add_64, convolution_11, squeeze_34, clone_7, div_7, convolution_12, squeeze_37, clone_8, div_8, convolution_13, squeeze_40, add_82, convolution_14, squeeze_43, clone_9, div_9, convolution_15, squeeze_46, clone_10, div_10, convolution_16, squeeze_49, add_100, convolution_17, squeeze_52, clone_11, div_11, convolution_18, squeeze_55, clone_12, div_12, mean, convolution_19, div_13, div_14, mul_147, convolution_21, squeeze_58, add_119, convolution_22, squeeze_61, clone_14, div_15, convolution_23, squeeze_64, clone_15, div_16, mean_1, convolution_24, div_17, div_18, mul_172, convolution_26, squeeze_67, add_139, convolution_27, squeeze_70, clone_17, div_19, convolution_28, squeeze_73, clone_18, div_20, mean_2, convolution_29, div_21, div_22, mul_197, convolution_31, squeeze_76, add_159, convolution_32, squeeze_79, clone_20, div_23, convolution_33, squeeze_82, clone_21, div_24, mean_3, convolution_34, div_25, div_26, mul_222, convolution_36, squeeze_85, add_179, convolution_37, squeeze_88, clone_23, div_27, convolution_38, squeeze_91, clone_24, div_28, mean_4, convolution_39, div_29, div_30, mul_247, convolution_41, squeeze_94, add_199, convolution_42, squeeze_97, clone_26, div_31, convolution_43, squeeze_100, clone_27, div_32, convolution_44, squeeze_103, add_216, convolution_45, squeeze_106, clone_28, div_33, convolution_46, squeeze_109, clone_29, div_34, convolution_47, squeeze_112, add_234, convolution_48, squeeze_115, clone_30, div_35, convolution_49, squeeze_118, clone_31, div_36, convolution_50, squeeze_121, add_252, convolution_51, squeeze_124, clone_32, div_37, convolution_52, squeeze_127, clone_33, div_38, convolution_53, squeeze_130, add_270, convolution_54, squeeze_133, clone_34, div_39, convolution_55, squeeze_136, clone_35, div_40, convolution_56, squeeze_139, add_288, convolution_57, squeeze_142, clone_36, div_41, convolution_58, squeeze_145, clone_37, div_42, mean_5, convolution_59, div_43, div_44, mul_387, convolution_61, squeeze_148, add_307, convolution_62, squeeze_151, clone_39, div_45, convolution_63, squeeze_154, clone_40, div_46, mean_6, convolution_64, div_47, div_48, mul_412, convolution_66, squeeze_157, add_327, convolution_67, squeeze_160, clone_42, div_49, convolution_68, squeeze_163, clone_43, div_50, mean_7, convolution_69, div_51, div_52, mul_437, convolution_71, squeeze_166, add_347, convolution_72, squeeze_169, clone_45, div_53, convolution_73, squeeze_172, clone_46, div_54, mean_8, convolution_74, div_55, div_56, mul_462, convolution_76, squeeze_175, add_367, convolution_77, squeeze_178, clone_48, div_57, convolution_78, squeeze_181, clone_49, div_58, mean_9, convolution_79, div_59, div_60, mul_487, convolution_81, squeeze_184, add_387, convolution_82, squeeze_187, clone_51, div_61, convolution_83, squeeze_190, clone_52, div_62, mean_10, convolution_84, div_63, div_64, mul_512, convolution_86, squeeze_193, add_407, convolution_87, squeeze_196, clone_54, div_65, convolution_88, squeeze_199, clone_55, div_66, mean_11, convolution_89, div_67, div_68, mul_537, convolution_91, squeeze_202, add_426, convolution_92, squeeze_205, clone_57, div_69, convolution_93, squeeze_208, clone_58, div_70, mean_12, convolution_94, div_71, div_72, mul_562, convolution_96, squeeze_211, add_446, convolution_97, squeeze_214, clone_60, div_73, convolution_98, squeeze_217, clone_61, div_74, mean_13, convolution_99, div_75, div_76, mul_587, convolution_101, squeeze_220, add_466, convolution_102, squeeze_223, clone_63, div_77, convolution_103, squeeze_226, clone_64, div_78, mean_14, convolution_104, div_79, div_80, mul_612, convolution_106, squeeze_229, add_486, convolution_107, squeeze_232, clone_66, div_81, convolution_108, squeeze_235, clone_67, div_82, mean_15, convolution_109, div_83, div_84, mul_637, convolution_111, squeeze_238, add_506, convolution_112, squeeze_241, clone_69, div_85, convolution_113, squeeze_244, clone_70, div_86, mean_16, convolution_114, div_87, div_88, mul_662, convolution_116, squeeze_247, add_526, convolution_117, squeeze_250, clone_72, div_89, convolution_118, squeeze_253, clone_73, div_90, mean_17, convolution_119, div_91, div_92, mul_687, convolution_121, squeeze_256, add_545, convolution_122, squeeze_259, clone_75, mean_18, convolution_123, view_1, permute_1, unsqueeze_350, unsqueeze_362, bitwise_and, unsqueeze_374, unsqueeze_386, unsqueeze_398, bitwise_and_1, unsqueeze_410, unsqueeze_422, unsqueeze_434, bitwise_and_2, unsqueeze_446, unsqueeze_458, unsqueeze_470, bitwise_and_3, unsqueeze_482, unsqueeze_494, unsqueeze_506, bitwise_and_4, unsqueeze_518, unsqueeze_530, unsqueeze_542, bitwise_and_5, unsqueeze_554, unsqueeze_566, unsqueeze_578, bitwise_and_6, unsqueeze_590, unsqueeze_602, unsqueeze_614, bitwise_and_7, unsqueeze_626, unsqueeze_638, unsqueeze_650, bitwise_and_8, unsqueeze_662, unsqueeze_674, unsqueeze_686, bitwise_and_9, unsqueeze_698, unsqueeze_710, unsqueeze_722, bitwise_and_10, unsqueeze_734, unsqueeze_746, unsqueeze_758, bitwise_and_11, unsqueeze_770, unsqueeze_782, unsqueeze_794, bitwise_and_12, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, bitwise_and_13, unsqueeze_1022, unsqueeze_1034, unsqueeze_1046, bitwise_and_14, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, bitwise_and_15, unsqueeze_1094, unsqueeze_1106, unsqueeze_1118, bitwise_and_16, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, bitwise_and_17, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, unsqueeze_1298, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1382, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_15, (24, ), (1, ))
    assert_size_stride(primals_17, (48, ), (1, ))
    assert_size_stride(primals_19, (48, ), (1, ))
    assert_size_stride(primals_21, (24, ), (1, ))
    assert_size_stride(primals_23, (48, ), (1, ))
    assert_size_stride(primals_25, (48, ), (1, ))
    assert_size_stride(primals_27, (24, ), (1, ))
    assert_size_stride(primals_29, (48, ), (1, ))
    assert_size_stride(primals_31, (48, ), (1, ))
    assert_size_stride(primals_33, (24, ), (1, ))
    assert_size_stride(primals_35, (120, ), (1, ))
    assert_size_stride(primals_37, (120, ), (1, ))
    assert_size_stride(primals_39, (40, ), (1, ))
    assert_size_stride(primals_41, (120, ), (1, ))
    assert_size_stride(primals_43, (120, ), (1, ))
    assert_size_stride(primals_45, (40, ), (1, ))
    assert_size_stride(primals_47, (120, ), (1, ))
    assert_size_stride(primals_49, (120, ), (1, ))
    assert_size_stride(primals_51, (40, ), (1, ))
    assert_size_stride(primals_53, (120, ), (1, ))
    assert_size_stride(primals_55, (120, ), (1, ))
    assert_size_stride(primals_57, (40, ), (1, ))
    assert_size_stride(primals_59, (120, ), (1, ))
    assert_size_stride(primals_61, (120, ), (1, ))
    assert_size_stride(primals_63, (40, ), (1, ))
    assert_size_stride(primals_65, (200, ), (1, ))
    assert_size_stride(primals_67, (200, ), (1, ))
    assert_size_stride(primals_69, (72, ), (1, ))
    assert_size_stride(primals_71, (216, ), (1, ))
    assert_size_stride(primals_73, (216, ), (1, ))
    assert_size_stride(primals_75, (72, ), (1, ))
    assert_size_stride(primals_77, (216, ), (1, ))
    assert_size_stride(primals_79, (216, ), (1, ))
    assert_size_stride(primals_81, (72, ), (1, ))
    assert_size_stride(primals_83, (216, ), (1, ))
    assert_size_stride(primals_85, (216, ), (1, ))
    assert_size_stride(primals_87, (72, ), (1, ))
    assert_size_stride(primals_89, (216, ), (1, ))
    assert_size_stride(primals_91, (216, ), (1, ))
    assert_size_stride(primals_93, (72, ), (1, ))
    assert_size_stride(primals_95, (360, ), (1, ))
    assert_size_stride(primals_97, (360, ), (1, ))
    assert_size_stride(primals_99, (120, ), (1, ))
    assert_size_stride(primals_101, (360, ), (1, ))
    assert_size_stride(primals_103, (360, ), (1, ))
    assert_size_stride(primals_105, (120, ), (1, ))
    assert_size_stride(primals_107, (360, ), (1, ))
    assert_size_stride(primals_109, (360, ), (1, ))
    assert_size_stride(primals_111, (120, ), (1, ))
    assert_size_stride(primals_113, (360, ), (1, ))
    assert_size_stride(primals_115, (360, ), (1, ))
    assert_size_stride(primals_117, (120, ), (1, ))
    assert_size_stride(primals_119, (360, ), (1, ))
    assert_size_stride(primals_121, (360, ), (1, ))
    assert_size_stride(primals_123, (120, ), (1, ))
    assert_size_stride(primals_125, (360, ), (1, ))
    assert_size_stride(primals_127, (360, ), (1, ))
    assert_size_stride(primals_129, (120, ), (1, ))
    assert_size_stride(primals_131, (720, ), (1, ))
    assert_size_stride(primals_133, (720, ), (1, ))
    assert_size_stride(primals_135, (184, ), (1, ))
    assert_size_stride(primals_137, (736, ), (1, ))
    assert_size_stride(primals_139, (736, ), (1, ))
    assert_size_stride(primals_141, (184, ), (1, ))
    assert_size_stride(primals_143, (736, ), (1, ))
    assert_size_stride(primals_145, (736, ), (1, ))
    assert_size_stride(primals_147, (184, ), (1, ))
    assert_size_stride(primals_149, (736, ), (1, ))
    assert_size_stride(primals_151, (736, ), (1, ))
    assert_size_stride(primals_153, (184, ), (1, ))
    assert_size_stride(primals_155, (736, ), (1, ))
    assert_size_stride(primals_157, (736, ), (1, ))
    assert_size_stride(primals_159, (184, ), (1, ))
    assert_size_stride(primals_161, (736, ), (1, ))
    assert_size_stride(primals_163, (736, ), (1, ))
    assert_size_stride(primals_165, (184, ), (1, ))
    assert_size_stride(primals_167, (1104, ), (1, ))
    assert_size_stride(primals_169, (1104, ), (1, ))
    assert_size_stride(primals_171, (224, ), (1, ))
    assert_size_stride(primals_173, (1344, ), (1, ))
    assert_size_stride(primals_177, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_178, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_179, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_180, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_181, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_182, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_183, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_184, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_185, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_186, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_187, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_188, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_189, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_190, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_191, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_192, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_193, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_194, (120, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_195, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (8, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_198, (120, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_200, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_201, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_202, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_203, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_205, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_207, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_208, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_209, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_210, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_212, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_214, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_215, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_216, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_217, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_219, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_221, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_222, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_223, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_224, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_226, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_228, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_229, (200, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_230, (200, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_231, (72, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_232, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_233, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_234, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_235, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_236, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_237, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_238, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_239, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_240, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_241, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_242, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_244, (360, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_245, (360, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_246, (24, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_248, (360, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_250, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_251, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_252, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_253, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_255, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_257, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_258, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_259, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_260, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_262, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_264, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_265, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_266, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_267, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_269, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_271, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_272, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_273, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_274, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_276, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_278, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_279, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_280, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_281, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_283, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_285, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_286, (720, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_287, (720, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_288, (32, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(primals_290, (720, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_292, (184, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(primals_293, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_294, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_295, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_297, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_299, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_300, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_301, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_302, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_304, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_306, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_307, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_308, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_309, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_311, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_313, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_314, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_315, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_316, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_318, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_320, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_321, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_322, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_323, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_325, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_327, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_328, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_329, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_330, (48, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_332, (1104, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_334, (224, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_335, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_336, (1984, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_598, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(clone, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_1, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_4, (16, ), (1, ))
    assert_size_stride(clone_1, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div_1, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_17, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_10, (16, ), (1, ))
    assert_size_stride(clone_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_4, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_13, (16, ), (1, ))
    assert_size_stride(add_29, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_5, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(clone_3, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(div_3, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_6, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(clone_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(div_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_7, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_22, (24, ), (1, ))
    assert_size_stride(add_46, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_8, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_25, (48, ), (1, ))
    assert_size_stride(clone_5, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_5, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_9, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_28, (48, ), (1, ))
    assert_size_stride(clone_6, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_6, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_10, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_31, (24, ), (1, ))
    assert_size_stride(add_64, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_11, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_34, (48, ), (1, ))
    assert_size_stride(clone_7, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_7, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_12, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_37, (48, ), (1, ))
    assert_size_stride(clone_8, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_8, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_13, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_40, (24, ), (1, ))
    assert_size_stride(add_82, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_14, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_43, (48, ), (1, ))
    assert_size_stride(clone_9, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_9, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_15, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_46, (48, ), (1, ))
    assert_size_stride(clone_10, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(div_10, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_16, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_49, (24, ), (1, ))
    assert_size_stride(add_100, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_17, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(squeeze_52, (120, ), (1, ))
    assert_size_stride(clone_11, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(div_11, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(convolution_18, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_55, (120, ), (1, ))
    assert_size_stride(clone_12, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_12, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_19, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(div_13, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(div_14, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_147, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_21, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_58, (40, ), (1, ))
    assert_size_stride(add_119, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_22, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_61, (120, ), (1, ))
    assert_size_stride(clone_14, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_15, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_23, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_64, (120, ), (1, ))
    assert_size_stride(clone_15, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_16, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_1, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_24, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_17, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_18, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_172, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_26, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_67, (40, ), (1, ))
    assert_size_stride(add_139, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_27, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_70, (120, ), (1, ))
    assert_size_stride(clone_17, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_19, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_28, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_73, (120, ), (1, ))
    assert_size_stride(clone_18, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_20, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_2, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_29, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_21, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_22, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_197, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_31, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_76, (40, ), (1, ))
    assert_size_stride(add_159, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_32, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_79, (120, ), (1, ))
    assert_size_stride(clone_20, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_23, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_33, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_82, (120, ), (1, ))
    assert_size_stride(clone_21, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_24, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_3, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_34, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_25, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_26, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_222, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_36, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_85, (40, ), (1, ))
    assert_size_stride(add_179, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_37, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_88, (120, ), (1, ))
    assert_size_stride(clone_23, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_27, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_38, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_91, (120, ), (1, ))
    assert_size_stride(clone_24, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(div_28, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_4, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(convolution_39, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_29, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(div_30, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_247, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_41, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_94, (40, ), (1, ))
    assert_size_stride(add_199, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_42, (8, 200, 28, 28), (156800, 1, 5600, 200))
    assert_size_stride(squeeze_97, (200, ), (1, ))
    assert_size_stride(clone_26, (8, 200, 28, 28), (156800, 1, 5600, 200))
    assert_size_stride(div_31, (8, 200, 28, 28), (156800, 1, 5600, 200))
    assert_size_stride(convolution_43, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(squeeze_100, (200, ), (1, ))
    assert_size_stride(clone_27, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_32, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_44, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_103, (72, ), (1, ))
    assert_size_stride(add_216, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_45, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_106, (216, ), (1, ))
    assert_size_stride(clone_28, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_33, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_46, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_109, (216, ), (1, ))
    assert_size_stride(clone_29, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_34, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_47, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_112, (72, ), (1, ))
    assert_size_stride(add_234, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_48, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_115, (216, ), (1, ))
    assert_size_stride(clone_30, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_35, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_49, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_118, (216, ), (1, ))
    assert_size_stride(clone_31, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_36, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_50, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_121, (72, ), (1, ))
    assert_size_stride(add_252, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_51, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_124, (216, ), (1, ))
    assert_size_stride(clone_32, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_37, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_52, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_127, (216, ), (1, ))
    assert_size_stride(clone_33, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_38, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_53, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_130, (72, ), (1, ))
    assert_size_stride(add_270, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_54, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_133, (216, ), (1, ))
    assert_size_stride(clone_34, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_39, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_55, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(squeeze_136, (216, ), (1, ))
    assert_size_stride(clone_35, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(div_40, (8, 216, 14, 14), (42336, 1, 3024, 216))
    assert_size_stride(convolution_56, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_139, (72, ), (1, ))
    assert_size_stride(add_288, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_57, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_142, (360, ), (1, ))
    assert_size_stride(clone_36, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_41, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_58, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_145, (360, ), (1, ))
    assert_size_stride(clone_37, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_42, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_5, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_59, (8, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(div_43, (8, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(div_44, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_387, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_61, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_148, (120, ), (1, ))
    assert_size_stride(add_307, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_62, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_151, (360, ), (1, ))
    assert_size_stride(clone_39, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_45, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_63, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_154, (360, ), (1, ))
    assert_size_stride(clone_40, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_46, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_6, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_64, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_47, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_48, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_412, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_66, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_157, (120, ), (1, ))
    assert_size_stride(add_327, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_67, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_160, (360, ), (1, ))
    assert_size_stride(clone_42, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_49, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_68, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_163, (360, ), (1, ))
    assert_size_stride(clone_43, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_50, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_7, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_69, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_51, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_52, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_437, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_71, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_166, (120, ), (1, ))
    assert_size_stride(add_347, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_72, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_169, (360, ), (1, ))
    assert_size_stride(clone_45, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_53, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_73, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_172, (360, ), (1, ))
    assert_size_stride(clone_46, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_54, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_8, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_74, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_55, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_56, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_462, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_76, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_175, (120, ), (1, ))
    assert_size_stride(add_367, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_77, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_178, (360, ), (1, ))
    assert_size_stride(clone_48, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_57, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_78, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_181, (360, ), (1, ))
    assert_size_stride(clone_49, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_58, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_9, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_79, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_59, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_60, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_487, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_81, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_184, (120, ), (1, ))
    assert_size_stride(add_387, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_82, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_187, (360, ), (1, ))
    assert_size_stride(clone_51, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_61, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_83, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(squeeze_190, (360, ), (1, ))
    assert_size_stride(clone_52, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(div_62, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(mean_10, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(convolution_84, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_63, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_64, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(mul_512, (8, 360, 14, 14), (70560, 1, 5040, 360))
    assert_size_stride(convolution_86, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(squeeze_193, (120, ), (1, ))
    assert_size_stride(add_407, (8, 120, 14, 14), (23520, 1, 1680, 120))
    assert_size_stride(convolution_87, (8, 720, 14, 14), (141120, 1, 10080, 720))
    assert_size_stride(squeeze_196, (720, ), (1, ))
    assert_size_stride(clone_54, (8, 720, 14, 14), (141120, 1, 10080, 720))
    assert_size_stride(div_65, (8, 720, 14, 14), (141120, 1, 10080, 720))
    assert_size_stride(convolution_88, (8, 720, 7, 7), (35280, 1, 5040, 720))
    assert_size_stride(squeeze_199, (720, ), (1, ))
    assert_size_stride(clone_55, (8, 720, 7, 7), (35280, 1, 5040, 720))
    assert_size_stride(div_66, (8, 720, 7, 7), (35280, 1, 5040, 720))
    assert_size_stride(mean_11, (8, 720, 1, 1), (720, 1, 720, 720))
    assert_size_stride(convolution_89, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_67, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_68, (8, 720, 1, 1), (720, 1, 720, 720))
    assert_size_stride(mul_537, (8, 720, 7, 7), (35280, 1, 5040, 720))
    assert_size_stride(convolution_91, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_202, (184, ), (1, ))
    assert_size_stride(add_426, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_92, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_205, (736, ), (1, ))
    assert_size_stride(clone_57, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_69, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_93, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_208, (736, ), (1, ))
    assert_size_stride(clone_58, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_70, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_12, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_94, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_71, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_72, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_562, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_96, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_211, (184, ), (1, ))
    assert_size_stride(add_446, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_97, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_214, (736, ), (1, ))
    assert_size_stride(clone_60, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_73, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_98, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_217, (736, ), (1, ))
    assert_size_stride(clone_61, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_74, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_13, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_99, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_75, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_76, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_587, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_101, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_220, (184, ), (1, ))
    assert_size_stride(add_466, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_102, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_223, (736, ), (1, ))
    assert_size_stride(clone_63, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_77, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_103, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_226, (736, ), (1, ))
    assert_size_stride(clone_64, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_78, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_14, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_104, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_79, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_80, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_612, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_106, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_229, (184, ), (1, ))
    assert_size_stride(add_486, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_107, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_232, (736, ), (1, ))
    assert_size_stride(clone_66, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_81, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_108, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_235, (736, ), (1, ))
    assert_size_stride(clone_67, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_82, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_15, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_109, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_83, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_84, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_637, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_111, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_238, (184, ), (1, ))
    assert_size_stride(add_506, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_112, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_241, (736, ), (1, ))
    assert_size_stride(clone_69, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_85, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_113, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(squeeze_244, (736, ), (1, ))
    assert_size_stride(clone_70, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(div_86, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(mean_16, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(convolution_114, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_87, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_88, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(mul_662, (8, 736, 7, 7), (36064, 1, 5152, 736))
    assert_size_stride(convolution_116, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(squeeze_247, (184, ), (1, ))
    assert_size_stride(add_526, (8, 184, 7, 7), (9016, 1, 1288, 184))
    assert_size_stride(convolution_117, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(squeeze_250, (1104, ), (1, ))
    assert_size_stride(clone_72, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(div_89, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(convolution_118, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(squeeze_253, (1104, ), (1, ))
    assert_size_stride(clone_73, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(div_90, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(mean_17, (8, 1104, 1, 1), (1104, 1, 1104, 1104))
    assert_size_stride(convolution_119, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_91, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(div_92, (8, 1104, 1, 1), (1104, 1, 1104, 1104))
    assert_size_stride(mul_687, (8, 1104, 7, 7), (54096, 1, 7728, 1104))
    assert_size_stride(convolution_121, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_256, (224, ), (1, ))
    assert_size_stride(add_545, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_122, (8, 1344, 7, 7), (65856, 1, 9408, 1344))
    assert_size_stride(squeeze_259, (1344, ), (1, ))
    assert_size_stride(clone_75, (8, 1344, 7, 7), (65856, 1, 9408, 1344))
    assert_size_stride(mean_18, (8, 1344, 1, 1), (1344, 1, 1344, 1344))
    assert_size_stride(convolution_123, (8, 1984, 1, 1), (1984, 1, 1984, 1984))
    assert_size_stride(view_1, (8, 1984), (1984, 1))
    assert_size_stride(permute_1, (1000, 1984), (1984, 1))
    assert_size_stride(unsqueeze_350, (1, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(bitwise_and, (8, 1104, 1, 1), (1104, 1, 1104, 1104))
    assert_size_stride(unsqueeze_374, (1, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(unsqueeze_398, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_1, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_410, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_422, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_2, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_446, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_470, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_3, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_482, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_494, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_4, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_518, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_530, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_542, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_5, (8, 736, 1, 1), (736, 1, 736, 736))
    assert_size_stride(unsqueeze_554, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_566, (1, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(unsqueeze_578, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(bitwise_and_6, (8, 720, 1, 1), (720, 1, 720, 720))
    assert_size_stride(unsqueeze_590, (1, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(unsqueeze_602, (1, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(unsqueeze_614, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_7, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_626, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_638, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_650, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_8, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_662, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_674, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_686, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_9, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_698, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_710, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_722, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_10, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_734, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_746, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_758, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_11, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_770, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_782, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_794, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(bitwise_and_12, (8, 360, 1, 1), (360, 1, 360, 360))
    assert_size_stride(unsqueeze_806, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_818, (1, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(unsqueeze_830, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_842, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_854, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_866, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_878, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_890, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_902, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_914, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_926, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_938, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_950, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_962, (1, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(unsqueeze_974, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_986, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_998, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1010, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_13, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1022, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1034, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1046, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_14, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1058, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1070, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1082, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_15, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1094, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1106, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1118, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_16, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1130, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1142, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1154, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_17, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_1166, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1178, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_1190, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1202, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1214, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1226, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1238, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1250, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1262, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1274, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1286, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(unsqueeze_1298, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1310, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1322, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1334, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1346, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1358, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1370, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1382, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1984), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1984), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view_1, out=buf1)
    del view_1
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = reinterpret_tensor(buf0, (8, 1984, 1, 1), (1984, 1, 1, 1), 0); del buf0  # reuse
    cpp_fused_convolution_backward_hardswish_backward_sum_0(c_void_p(buf3.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convolution_123.data_ptr()), c_void_p(buf2.data_ptr()))
    del convolution_123
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf4 = aten.convolution_backward(buf3, mean_18, primals_336, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf3
    del mean_18
    del primals_336
    buf5 = buf4[0]
    buf6 = buf4[1]
    del buf4
    buf7 = empty((1344, ), device='cpu', dtype=torch.float32)
    buf8 = empty((1344, ), device='cpu', dtype=torch.float32)
    buf9 = empty((1344, ), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((8, 1344, 7, 7), (65856, 1, 9408, 1344), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_1(c_void_p(clone_75.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(convolution_122.data_ptr()), c_void_p(unsqueeze_350.data_ptr()), c_void_p(squeeze_259.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf5
    del buf8
    del clone_75
    del convolution_122
    del primals_173
    del squeeze_259
    del unsqueeze_350
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf11 = aten.convolution_backward(buf10, add_545, primals_335, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_545
    del buf10
    del primals_335
    buf12 = buf11[0]
    buf13 = buf11[1]
    del buf11
    buf14 = empty((224, ), device='cpu', dtype=torch.float32)
    buf15 = empty((224, ), device='cpu', dtype=torch.float32)
    buf16 = empty((224, ), device='cpu', dtype=torch.float32)
    buf17 = buf12; del buf12  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_2(c_void_p(buf17.data_ptr()), c_void_p(convolution_121.data_ptr()), c_void_p(unsqueeze_362.data_ptr()), c_void_p(squeeze_256.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    del buf15
    del convolution_121
    del primals_171
    del squeeze_256
    del unsqueeze_362
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf18 = aten.convolution_backward(buf17, mul_687, primals_334, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf17
    del mul_687
    del primals_334
    buf19 = buf18[0]
    buf20 = buf18[1]
    del buf18
    buf21 = empty_strided((8, 1104, 1, 1), (1104, 1, 8832, 8832), device='cpu', dtype=torch.float32)
    buf22 = reinterpret_tensor(buf21, (8, 1104, 1, 1), (1104, 1, 1104, 1104), 0); del buf21  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_3(c_void_p(buf22.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(div_90.data_ptr()), c_void_p(bitwise_and.data_ptr()))
    del bitwise_and
    del div_90
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf23 = aten.convolution_backward(buf22, div_91, primals_332, [1104], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf22
    del div_91
    del primals_332
    buf24 = buf23[0]
    buf25 = buf23[1]
    buf26 = buf23[2]
    del buf23
    buf27 = reinterpret_tensor(buf24, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf24  # reuse
    cpp_fused_convolution_backward_hardswish_backward_4(c_void_p(buf27.data_ptr()), c_void_p(convolution_119.data_ptr()))
    del convolution_119
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf28 = aten.convolution_backward(buf27, mean_17, primals_330, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf27
    del mean_17
    del primals_330
    buf29 = buf28[0]
    buf30 = buf28[1]
    buf31 = buf28[2]
    del buf28
    buf32 = empty((1104, ), device='cpu', dtype=torch.float32)
    buf33 = empty((1104, ), device='cpu', dtype=torch.float32)
    buf34 = buf19; del buf19  # reuse
    buf35 = buf33; del buf33  # reuse
    buf36 = buf34; del buf34  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_5(c_void_p(buf36.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(clone_73.data_ptr()), c_void_p(div_92.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(convolution_118.data_ptr()), c_void_p(unsqueeze_374.data_ptr()), c_void_p(squeeze_253.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf32.data_ptr()))
    del buf29
    del clone_73
    del convolution_118
    del div_92
    del primals_169
    del squeeze_253
    del unsqueeze_374
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf37 = aten.convolution_backward(buf36, div_89, primals_329, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False])
    del buf36
    del div_89
    del primals_329
    buf38 = buf37[0]
    buf39 = buf37[1]
    del buf37
    buf40 = empty((1104, ), device='cpu', dtype=torch.float32)
    buf41 = empty((1104, ), device='cpu', dtype=torch.float32)
    buf42 = empty((1104, ), device='cpu', dtype=torch.float32)
    buf43 = buf38; del buf38  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_6(c_void_p(buf43.data_ptr()), c_void_p(clone_72.data_ptr()), c_void_p(convolution_117.data_ptr()), c_void_p(unsqueeze_386.data_ptr()), c_void_p(squeeze_250.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()))
    del buf41
    del clone_72
    del convolution_117
    del primals_167
    del squeeze_250
    del unsqueeze_386
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf44 = aten.convolution_backward(buf43, add_526, primals_328, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_526
    del buf43
    del primals_328
    buf45 = buf44[0]
    buf46 = buf44[1]
    del buf44
    buf47 = empty((184, ), device='cpu', dtype=torch.float32)
    buf48 = empty((184, ), device='cpu', dtype=torch.float32)
    buf49 = empty((184, ), device='cpu', dtype=torch.float32)
    buf50 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_7(c_void_p(buf45.data_ptr()), c_void_p(convolution_116.data_ptr()), c_void_p(unsqueeze_398.data_ptr()), c_void_p(squeeze_247.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    del convolution_116
    del primals_165
    del squeeze_247
    del unsqueeze_398
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf51 = aten.convolution_backward(buf50, mul_662, primals_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_662
    del primals_327
    buf52 = buf51[0]
    buf53 = buf51[1]
    del buf51
    buf54 = empty_strided((8, 736, 1, 1), (736, 1, 5888, 5888), device='cpu', dtype=torch.float32)
    buf55 = reinterpret_tensor(buf54, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf54  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_8(c_void_p(buf55.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(div_86.data_ptr()), c_void_p(bitwise_and_1.data_ptr()))
    del bitwise_and_1
    del div_86
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf56 = aten.convolution_backward(buf55, div_87, primals_325, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf55
    del div_87
    del primals_325
    buf57 = buf56[0]
    buf58 = buf56[1]
    buf59 = buf56[2]
    del buf56
    buf60 = reinterpret_tensor(buf57, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf57  # reuse
    cpp_fused_convolution_backward_hardswish_backward_9(c_void_p(buf60.data_ptr()), c_void_p(convolution_114.data_ptr()))
    del convolution_114
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf61 = aten.convolution_backward(buf60, mean_16, primals_323, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf60
    del mean_16
    del primals_323
    buf62 = buf61[0]
    buf63 = buf61[1]
    buf64 = buf61[2]
    del buf61
    buf65 = empty((736, ), device='cpu', dtype=torch.float32)
    buf66 = empty((736, ), device='cpu', dtype=torch.float32)
    buf67 = buf52; del buf52  # reuse
    buf68 = buf66; del buf66  # reuse
    buf69 = buf67; del buf67  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_10(c_void_p(buf69.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(clone_70.data_ptr()), c_void_p(div_88.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(convolution_113.data_ptr()), c_void_p(unsqueeze_410.data_ptr()), c_void_p(squeeze_244.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf65.data_ptr()))
    del clone_70
    del convolution_113
    del div_88
    del primals_163
    del squeeze_244
    del unsqueeze_410
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf70 = aten.convolution_backward(buf69, div_85, primals_322, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
    del buf69
    del div_85
    del primals_322
    buf71 = buf70[0]
    buf72 = buf70[1]
    del buf70
    buf73 = empty((736, ), device='cpu', dtype=torch.float32)
    buf74 = empty((736, ), device='cpu', dtype=torch.float32)
    buf75 = empty((736, ), device='cpu', dtype=torch.float32)
    buf76 = buf71; del buf71  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_11(c_void_p(buf76.data_ptr()), c_void_p(clone_69.data_ptr()), c_void_p(convolution_112.data_ptr()), c_void_p(unsqueeze_422.data_ptr()), c_void_p(squeeze_241.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del clone_69
    del convolution_112
    del primals_161
    del squeeze_241
    del unsqueeze_422
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf77 = aten.convolution_backward(buf76, add_506, primals_321, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_506
    del buf76
    del primals_321
    buf78 = buf77[0]
    buf79 = buf77[1]
    del buf77
    buf80 = buf48; del buf48  # reuse
    buf81 = empty((184, ), device='cpu', dtype=torch.float32)
    buf82 = empty((184, ), device='cpu', dtype=torch.float32)
    buf83 = buf50; del buf50  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_12(c_void_p(buf45.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(convolution_111.data_ptr()), c_void_p(unsqueeze_434.data_ptr()), c_void_p(squeeze_238.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del convolution_111
    del primals_159
    del squeeze_238
    del unsqueeze_434
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf84 = aten.convolution_backward(buf83, mul_637, primals_320, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_637
    del primals_320
    buf85 = buf84[0]
    buf86 = buf84[1]
    del buf84
    buf87 = reinterpret_tensor(buf62, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf62  # reuse
    buf88 = reinterpret_tensor(buf87, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf87  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_13(c_void_p(buf88.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(div_82.data_ptr()), c_void_p(bitwise_and_2.data_ptr()))
    del bitwise_and_2
    del div_82
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf89 = aten.convolution_backward(buf88, div_83, primals_318, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf88
    del div_83
    del primals_318
    buf90 = buf89[0]
    buf91 = buf89[1]
    buf92 = buf89[2]
    del buf89
    buf93 = reinterpret_tensor(buf90, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf90  # reuse
    cpp_fused_convolution_backward_hardswish_backward_14(c_void_p(buf93.data_ptr()), c_void_p(convolution_109.data_ptr()))
    del convolution_109
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf94 = aten.convolution_backward(buf93, mean_15, primals_316, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf93
    del mean_15
    del primals_316
    buf95 = buf94[0]
    buf96 = buf94[1]
    buf97 = buf94[2]
    del buf94
    buf98 = buf74; del buf74  # reuse
    buf99 = empty((736, ), device='cpu', dtype=torch.float32)
    buf100 = buf85; del buf85  # reuse
    buf101 = buf99; del buf99  # reuse
    buf102 = buf100; del buf100  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_15(c_void_p(buf102.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(clone_67.data_ptr()), c_void_p(div_84.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_108.data_ptr()), c_void_p(unsqueeze_446.data_ptr()), c_void_p(squeeze_235.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(buf98.data_ptr()))
    del clone_67
    del convolution_108
    del div_84
    del primals_157
    del squeeze_235
    del unsqueeze_446
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf103 = aten.convolution_backward(buf102, div_81, primals_315, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
    del buf102
    del div_81
    del primals_315
    buf104 = buf103[0]
    buf105 = buf103[1]
    del buf103
    buf106 = empty((736, ), device='cpu', dtype=torch.float32)
    buf107 = empty((736, ), device='cpu', dtype=torch.float32)
    buf108 = empty((736, ), device='cpu', dtype=torch.float32)
    buf109 = buf104; del buf104  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_16(c_void_p(buf109.data_ptr()), c_void_p(clone_66.data_ptr()), c_void_p(convolution_107.data_ptr()), c_void_p(unsqueeze_458.data_ptr()), c_void_p(squeeze_232.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()))
    del clone_66
    del convolution_107
    del primals_155
    del squeeze_232
    del unsqueeze_458
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf110 = aten.convolution_backward(buf109, add_486, primals_314, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_486
    del buf109
    del primals_314
    buf111 = buf110[0]
    buf112 = buf110[1]
    del buf110
    buf113 = buf81; del buf81  # reuse
    buf114 = empty((184, ), device='cpu', dtype=torch.float32)
    buf115 = buf83; del buf83  # reuse
    buf116 = buf114; del buf114  # reuse
    cpp_fused_add_native_batch_norm_backward_17(c_void_p(buf116.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(convolution_106.data_ptr()), c_void_p(unsqueeze_470.data_ptr()), c_void_p(squeeze_229.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()))
    del convolution_106
    del primals_153
    del squeeze_229
    del unsqueeze_470
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf117 = aten.convolution_backward(buf115, mul_612, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_612
    del primals_313
    buf118 = buf117[0]
    buf119 = buf117[1]
    del buf117
    buf120 = reinterpret_tensor(buf95, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf95  # reuse
    buf121 = reinterpret_tensor(buf120, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf120  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_18(c_void_p(buf121.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(div_78.data_ptr()), c_void_p(bitwise_and_3.data_ptr()))
    del bitwise_and_3
    del div_78
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf122 = aten.convolution_backward(buf121, div_79, primals_311, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf121
    del div_79
    del primals_311
    buf123 = buf122[0]
    buf124 = buf122[1]
    buf125 = buf122[2]
    del buf122
    buf126 = reinterpret_tensor(buf123, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf123  # reuse
    cpp_fused_convolution_backward_hardswish_backward_19(c_void_p(buf126.data_ptr()), c_void_p(convolution_104.data_ptr()))
    del convolution_104
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf127 = aten.convolution_backward(buf126, mean_14, primals_309, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf126
    del mean_14
    del primals_309
    buf128 = buf127[0]
    buf129 = buf127[1]
    buf130 = buf127[2]
    del buf127
    buf131 = buf107; del buf107  # reuse
    buf132 = empty((736, ), device='cpu', dtype=torch.float32)
    buf133 = buf118; del buf118  # reuse
    buf134 = buf132; del buf132  # reuse
    buf135 = buf133; del buf133  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_20(c_void_p(buf135.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(clone_64.data_ptr()), c_void_p(div_80.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(convolution_103.data_ptr()), c_void_p(unsqueeze_482.data_ptr()), c_void_p(squeeze_226.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf131.data_ptr()))
    del clone_64
    del convolution_103
    del div_80
    del primals_151
    del squeeze_226
    del unsqueeze_482
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf136 = aten.convolution_backward(buf135, div_77, primals_308, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
    del buf135
    del div_77
    del primals_308
    buf137 = buf136[0]
    buf138 = buf136[1]
    del buf136
    buf139 = empty((736, ), device='cpu', dtype=torch.float32)
    buf140 = empty((736, ), device='cpu', dtype=torch.float32)
    buf141 = empty((736, ), device='cpu', dtype=torch.float32)
    buf142 = buf137; del buf137  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_21(c_void_p(buf142.data_ptr()), c_void_p(clone_63.data_ptr()), c_void_p(convolution_102.data_ptr()), c_void_p(unsqueeze_494.data_ptr()), c_void_p(squeeze_223.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    del clone_63
    del convolution_102
    del primals_149
    del squeeze_223
    del unsqueeze_494
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf143 = aten.convolution_backward(buf142, add_466, primals_307, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_466
    del buf142
    del primals_307
    buf144 = buf143[0]
    buf145 = buf143[1]
    del buf143
    buf146 = empty((184, ), device='cpu', dtype=torch.float32)
    buf147 = empty((184, ), device='cpu', dtype=torch.float32)
    buf148 = buf115; del buf115  # reuse
    buf150 = buf148; del buf148  # reuse
    buf149 = buf147; del buf147  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_22(c_void_p(buf150.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(convolution_101.data_ptr()), c_void_p(unsqueeze_506.data_ptr()), c_void_p(squeeze_220.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf146.data_ptr()))
    del convolution_101
    del primals_147
    del squeeze_220
    del unsqueeze_506
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf151 = aten.convolution_backward(buf150, mul_587, primals_306, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_587
    del primals_306
    buf152 = buf151[0]
    buf153 = buf151[1]
    del buf151
    buf154 = reinterpret_tensor(buf128, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf128  # reuse
    buf155 = reinterpret_tensor(buf154, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf154  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_23(c_void_p(buf155.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(div_74.data_ptr()), c_void_p(bitwise_and_4.data_ptr()))
    del bitwise_and_4
    del div_74
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf156 = aten.convolution_backward(buf155, div_75, primals_304, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf155
    del div_75
    del primals_304
    buf157 = buf156[0]
    buf158 = buf156[1]
    buf159 = buf156[2]
    del buf156
    buf160 = reinterpret_tensor(buf157, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf157  # reuse
    cpp_fused_convolution_backward_hardswish_backward_24(c_void_p(buf160.data_ptr()), c_void_p(convolution_99.data_ptr()))
    del convolution_99
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf161 = aten.convolution_backward(buf160, mean_13, primals_302, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf160
    del mean_13
    del primals_302
    buf162 = buf161[0]
    buf163 = buf161[1]
    buf164 = buf161[2]
    del buf161
    buf165 = buf140; del buf140  # reuse
    buf166 = empty((736, ), device='cpu', dtype=torch.float32)
    buf167 = buf152; del buf152  # reuse
    buf168 = buf166; del buf166  # reuse
    buf169 = buf167; del buf167  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_25(c_void_p(buf169.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(clone_61.data_ptr()), c_void_p(div_76.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(convolution_98.data_ptr()), c_void_p(unsqueeze_518.data_ptr()), c_void_p(squeeze_217.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(buf165.data_ptr()))
    del clone_61
    del convolution_98
    del div_76
    del primals_145
    del squeeze_217
    del unsqueeze_518
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf170 = aten.convolution_backward(buf169, div_73, primals_301, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
    del buf169
    del div_73
    del primals_301
    buf171 = buf170[0]
    buf172 = buf170[1]
    del buf170
    buf173 = empty((736, ), device='cpu', dtype=torch.float32)
    buf174 = empty((736, ), device='cpu', dtype=torch.float32)
    buf175 = empty((736, ), device='cpu', dtype=torch.float32)
    buf176 = buf171; del buf171  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_26(c_void_p(buf176.data_ptr()), c_void_p(clone_60.data_ptr()), c_void_p(convolution_97.data_ptr()), c_void_p(unsqueeze_530.data_ptr()), c_void_p(squeeze_214.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    del clone_60
    del convolution_97
    del primals_143
    del squeeze_214
    del unsqueeze_530
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf177 = aten.convolution_backward(buf176, add_446, primals_300, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_446
    del buf176
    del primals_300
    buf178 = buf177[0]
    buf179 = buf177[1]
    del buf177
    buf180 = buf111; del buf111  # reuse
    buf181 = empty((184, ), device='cpu', dtype=torch.float32)
    buf182 = empty((184, ), device='cpu', dtype=torch.float32)
    buf183 = empty((184, ), device='cpu', dtype=torch.float32)
    buf184 = buf150; del buf150  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_27(c_void_p(buf180.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(convolution_96.data_ptr()), c_void_p(unsqueeze_542.data_ptr()), c_void_p(squeeze_211.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    del buf144
    del buf178
    del buf45
    del buf78
    del convolution_96
    del primals_141
    del squeeze_211
    del unsqueeze_542
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf185 = aten.convolution_backward(buf184, mul_562, primals_299, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf184
    del mul_562
    del primals_299
    buf186 = buf185[0]
    buf187 = buf185[1]
    del buf185
    buf188 = reinterpret_tensor(buf162, (8, 736, 1, 1), (736, 1, 5888, 5888), 0); del buf162  # reuse
    buf189 = reinterpret_tensor(buf188, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf188  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_28(c_void_p(buf189.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(div_70.data_ptr()), c_void_p(bitwise_and_5.data_ptr()))
    del bitwise_and_5
    del div_70
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf190 = aten.convolution_backward(buf189, div_71, primals_297, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf189
    del div_71
    del primals_297
    buf191 = buf190[0]
    buf192 = buf190[1]
    buf193 = buf190[2]
    del buf190
    buf194 = reinterpret_tensor(buf191, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf191  # reuse
    cpp_fused_convolution_backward_hardswish_backward_29(c_void_p(buf194.data_ptr()), c_void_p(convolution_94.data_ptr()))
    del convolution_94
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf195 = aten.convolution_backward(buf194, mean_12, primals_295, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf194
    del mean_12
    del primals_295
    buf196 = buf195[0]
    buf197 = buf195[1]
    buf198 = buf195[2]
    del buf195
    buf199 = buf174; del buf174  # reuse
    buf200 = empty((736, ), device='cpu', dtype=torch.float32)
    buf201 = buf186; del buf186  # reuse
    buf202 = buf200; del buf200  # reuse
    buf203 = buf201; del buf201  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_30(c_void_p(buf203.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(clone_58.data_ptr()), c_void_p(div_72.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(convolution_93.data_ptr()), c_void_p(unsqueeze_554.data_ptr()), c_void_p(squeeze_208.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(buf199.data_ptr()))
    del buf196
    del clone_58
    del convolution_93
    del div_72
    del primals_139
    del squeeze_208
    del unsqueeze_554
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf204 = aten.convolution_backward(buf203, div_69, primals_294, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False])
    del buf203
    del div_69
    del primals_294
    buf205 = buf204[0]
    buf206 = buf204[1]
    del buf204
    buf207 = empty((736, ), device='cpu', dtype=torch.float32)
    buf208 = empty((736, ), device='cpu', dtype=torch.float32)
    buf209 = empty((736, ), device='cpu', dtype=torch.float32)
    buf210 = buf205; del buf205  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31(c_void_p(buf210.data_ptr()), c_void_p(clone_57.data_ptr()), c_void_p(convolution_92.data_ptr()), c_void_p(unsqueeze_566.data_ptr()), c_void_p(squeeze_205.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    del buf208
    del clone_57
    del convolution_92
    del primals_137
    del squeeze_205
    del unsqueeze_566
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf211 = aten.convolution_backward(buf210, add_426, primals_293, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_426
    del buf210
    del primals_293
    buf212 = buf211[0]
    buf213 = buf211[1]
    del buf211
    buf214 = buf182; del buf182  # reuse
    buf215 = empty((184, ), device='cpu', dtype=torch.float32)
    buf216 = empty((184, ), device='cpu', dtype=torch.float32)
    buf217 = buf180; del buf180  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_32(c_void_p(buf217.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(convolution_91.data_ptr()), c_void_p(unsqueeze_578.data_ptr()), c_void_p(squeeze_202.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del buf212
    del buf215
    del convolution_91
    del primals_135
    del squeeze_202
    del unsqueeze_578
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf218 = aten.convolution_backward(buf217, mul_537, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf217
    del mul_537
    del primals_292
    buf219 = buf218[0]
    buf220 = buf218[1]
    del buf218
    buf221 = empty_strided((8, 720, 1, 1), (720, 1, 5760, 5760), device='cpu', dtype=torch.float32)
    buf222 = reinterpret_tensor(buf221, (8, 720, 1, 1), (720, 1, 720, 720), 0); del buf221  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_33(c_void_p(buf222.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(div_66.data_ptr()), c_void_p(bitwise_and_6.data_ptr()))
    del bitwise_and_6
    del div_66
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf223 = aten.convolution_backward(buf222, div_67, primals_290, [720], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf222
    del div_67
    del primals_290
    buf224 = buf223[0]
    buf225 = buf223[1]
    buf226 = buf223[2]
    del buf223
    buf227 = reinterpret_tensor(buf224, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf224  # reuse
    cpp_fused_convolution_backward_hardswish_backward_34(c_void_p(buf227.data_ptr()), c_void_p(convolution_89.data_ptr()))
    del convolution_89
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf228 = aten.convolution_backward(buf227, mean_11, primals_288, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf227
    del mean_11
    del primals_288
    buf229 = buf228[0]
    buf230 = buf228[1]
    buf231 = buf228[2]
    del buf228
    buf232 = empty((720, ), device='cpu', dtype=torch.float32)
    buf233 = empty((720, ), device='cpu', dtype=torch.float32)
    buf234 = buf219; del buf219  # reuse
    buf235 = buf233; del buf233  # reuse
    buf236 = buf234; del buf234  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_35(c_void_p(buf236.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(clone_55.data_ptr()), c_void_p(div_68.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(convolution_88.data_ptr()), c_void_p(unsqueeze_590.data_ptr()), c_void_p(squeeze_199.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf232.data_ptr()))
    del buf229
    del clone_55
    del convolution_88
    del div_68
    del primals_133
    del squeeze_199
    del unsqueeze_590
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf237 = aten.convolution_backward(buf236, div_65, primals_287, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 720, [True, True, False])
    del buf236
    del div_65
    del primals_287
    buf238 = buf237[0]
    buf239 = buf237[1]
    del buf237
    buf240 = empty((720, ), device='cpu', dtype=torch.float32)
    buf241 = empty((720, ), device='cpu', dtype=torch.float32)
    buf242 = empty((720, ), device='cpu', dtype=torch.float32)
    buf243 = buf238; del buf238  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_36(c_void_p(buf243.data_ptr()), c_void_p(clone_54.data_ptr()), c_void_p(convolution_87.data_ptr()), c_void_p(unsqueeze_602.data_ptr()), c_void_p(squeeze_196.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    del buf241
    del clone_54
    del convolution_87
    del primals_131
    del squeeze_196
    del unsqueeze_602
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf244 = aten.convolution_backward(buf243, add_407, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_407
    del buf243
    del primals_286
    buf245 = buf244[0]
    buf246 = buf244[1]
    del buf244
    buf247 = empty((120, ), device='cpu', dtype=torch.float32)
    buf248 = empty((120, ), device='cpu', dtype=torch.float32)
    buf249 = empty((120, ), device='cpu', dtype=torch.float32)
    buf250 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_37(c_void_p(buf245.data_ptr()), c_void_p(convolution_86.data_ptr()), c_void_p(unsqueeze_614.data_ptr()), c_void_p(squeeze_193.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()))
    del convolution_86
    del primals_129
    del squeeze_193
    del unsqueeze_614
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf251 = aten.convolution_backward(buf250, mul_512, primals_285, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_512
    del primals_285
    buf252 = buf251[0]
    buf253 = buf251[1]
    del buf251
    buf254 = empty_strided((8, 360, 1, 1), (360, 1, 2880, 2880), device='cpu', dtype=torch.float32)
    buf255 = reinterpret_tensor(buf254, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf254  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_38(c_void_p(buf255.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(div_62.data_ptr()), c_void_p(bitwise_and_7.data_ptr()))
    del bitwise_and_7
    del div_62
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf256 = aten.convolution_backward(buf255, div_63, primals_283, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf255
    del div_63
    del primals_283
    buf257 = buf256[0]
    buf258 = buf256[1]
    buf259 = buf256[2]
    del buf256
    buf260 = reinterpret_tensor(buf257, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf257  # reuse
    cpp_fused_convolution_backward_hardswish_backward_39(c_void_p(buf260.data_ptr()), c_void_p(convolution_84.data_ptr()))
    del convolution_84
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf261 = aten.convolution_backward(buf260, mean_10, primals_281, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf260
    del mean_10
    del primals_281
    buf262 = buf261[0]
    buf263 = buf261[1]
    buf264 = buf261[2]
    del buf261
    buf265 = empty((360, ), device='cpu', dtype=torch.float32)
    buf266 = empty((360, ), device='cpu', dtype=torch.float32)
    buf267 = buf252; del buf252  # reuse
    buf268 = buf266; del buf266  # reuse
    buf269 = buf267; del buf267  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_40(c_void_p(buf269.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(clone_52.data_ptr()), c_void_p(div_64.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(convolution_83.data_ptr()), c_void_p(unsqueeze_626.data_ptr()), c_void_p(squeeze_190.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf265.data_ptr()))
    del clone_52
    del convolution_83
    del div_64
    del primals_127
    del squeeze_190
    del unsqueeze_626
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf270 = aten.convolution_backward(buf269, div_61, primals_280, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
    del buf269
    del div_61
    del primals_280
    buf271 = buf270[0]
    buf272 = buf270[1]
    del buf270
    buf273 = empty((360, ), device='cpu', dtype=torch.float32)
    buf274 = empty((360, ), device='cpu', dtype=torch.float32)
    buf275 = empty((360, ), device='cpu', dtype=torch.float32)
    buf276 = buf271; del buf271  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_41(c_void_p(buf276.data_ptr()), c_void_p(clone_51.data_ptr()), c_void_p(convolution_82.data_ptr()), c_void_p(unsqueeze_638.data_ptr()), c_void_p(squeeze_187.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()))
    del clone_51
    del convolution_82
    del primals_125
    del squeeze_187
    del unsqueeze_638
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf277 = aten.convolution_backward(buf276, add_387, primals_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_387
    del buf276
    del primals_279
    buf278 = buf277[0]
    buf279 = buf277[1]
    del buf277
    buf280 = buf248; del buf248  # reuse
    buf281 = empty((120, ), device='cpu', dtype=torch.float32)
    buf282 = empty((120, ), device='cpu', dtype=torch.float32)
    buf283 = buf250; del buf250  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_42(c_void_p(buf245.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(convolution_81.data_ptr()), c_void_p(unsqueeze_650.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    del convolution_81
    del primals_123
    del squeeze_184
    del unsqueeze_650
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf284 = aten.convolution_backward(buf283, mul_487, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_487
    del primals_278
    buf285 = buf284[0]
    buf286 = buf284[1]
    del buf284
    buf287 = reinterpret_tensor(buf262, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf262  # reuse
    buf288 = reinterpret_tensor(buf287, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf287  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_43(c_void_p(buf288.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(bitwise_and_8.data_ptr()))
    del bitwise_and_8
    del div_58
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf289 = aten.convolution_backward(buf288, div_59, primals_276, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf288
    del div_59
    del primals_276
    buf290 = buf289[0]
    buf291 = buf289[1]
    buf292 = buf289[2]
    del buf289
    buf293 = reinterpret_tensor(buf290, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf290  # reuse
    cpp_fused_convolution_backward_hardswish_backward_44(c_void_p(buf293.data_ptr()), c_void_p(convolution_79.data_ptr()))
    del convolution_79
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf294 = aten.convolution_backward(buf293, mean_9, primals_274, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf293
    del mean_9
    del primals_274
    buf295 = buf294[0]
    buf296 = buf294[1]
    buf297 = buf294[2]
    del buf294
    buf298 = buf274; del buf274  # reuse
    buf299 = empty((360, ), device='cpu', dtype=torch.float32)
    buf300 = buf285; del buf285  # reuse
    buf301 = buf299; del buf299  # reuse
    buf302 = buf300; del buf300  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_45(c_void_p(buf302.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(clone_49.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(convolution_78.data_ptr()), c_void_p(unsqueeze_662.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf298.data_ptr()))
    del clone_49
    del convolution_78
    del div_60
    del primals_121
    del squeeze_181
    del unsqueeze_662
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf303 = aten.convolution_backward(buf302, div_57, primals_273, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
    del buf302
    del div_57
    del primals_273
    buf304 = buf303[0]
    buf305 = buf303[1]
    del buf303
    buf306 = empty((360, ), device='cpu', dtype=torch.float32)
    buf307 = empty((360, ), device='cpu', dtype=torch.float32)
    buf308 = empty((360, ), device='cpu', dtype=torch.float32)
    buf309 = buf304; del buf304  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_46(c_void_p(buf309.data_ptr()), c_void_p(clone_48.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(unsqueeze_674.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    del clone_48
    del convolution_77
    del primals_119
    del squeeze_178
    del unsqueeze_674
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf310 = aten.convolution_backward(buf309, add_367, primals_272, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_367
    del buf309
    del primals_272
    buf311 = buf310[0]
    buf312 = buf310[1]
    del buf310
    buf313 = buf281; del buf281  # reuse
    buf314 = empty((120, ), device='cpu', dtype=torch.float32)
    buf315 = buf283; del buf283  # reuse
    buf316 = buf314; del buf314  # reuse
    cpp_fused_add_native_batch_norm_backward_47(c_void_p(buf316.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(unsqueeze_686.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf315.data_ptr()))
    del convolution_76
    del primals_117
    del squeeze_175
    del unsqueeze_686
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf317 = aten.convolution_backward(buf315, mul_462, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_462
    del primals_271
    buf318 = buf317[0]
    buf319 = buf317[1]
    del buf317
    buf320 = reinterpret_tensor(buf295, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf295  # reuse
    buf321 = reinterpret_tensor(buf320, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf320  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_48(c_void_p(buf321.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(bitwise_and_9.data_ptr()))
    del bitwise_and_9
    del div_54
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf322 = aten.convolution_backward(buf321, div_55, primals_269, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf321
    del div_55
    del primals_269
    buf323 = buf322[0]
    buf324 = buf322[1]
    buf325 = buf322[2]
    del buf322
    buf326 = reinterpret_tensor(buf323, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf323  # reuse
    cpp_fused_convolution_backward_hardswish_backward_49(c_void_p(buf326.data_ptr()), c_void_p(convolution_74.data_ptr()))
    del convolution_74
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf327 = aten.convolution_backward(buf326, mean_8, primals_267, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf326
    del mean_8
    del primals_267
    buf328 = buf327[0]
    buf329 = buf327[1]
    buf330 = buf327[2]
    del buf327
    buf331 = buf307; del buf307  # reuse
    buf332 = empty((360, ), device='cpu', dtype=torch.float32)
    buf333 = buf318; del buf318  # reuse
    buf334 = buf332; del buf332  # reuse
    buf335 = buf333; del buf333  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_50(c_void_p(buf335.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(clone_46.data_ptr()), c_void_p(div_56.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(unsqueeze_698.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf331.data_ptr()))
    del clone_46
    del convolution_73
    del div_56
    del primals_115
    del squeeze_172
    del unsqueeze_698
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf336 = aten.convolution_backward(buf335, div_53, primals_266, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
    del buf335
    del div_53
    del primals_266
    buf337 = buf336[0]
    buf338 = buf336[1]
    del buf336
    buf339 = empty((360, ), device='cpu', dtype=torch.float32)
    buf340 = empty((360, ), device='cpu', dtype=torch.float32)
    buf341 = empty((360, ), device='cpu', dtype=torch.float32)
    buf342 = buf337; del buf337  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_51(c_void_p(buf342.data_ptr()), c_void_p(clone_45.data_ptr()), c_void_p(convolution_72.data_ptr()), c_void_p(unsqueeze_710.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()))
    del clone_45
    del convolution_72
    del primals_113
    del squeeze_169
    del unsqueeze_710
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf343 = aten.convolution_backward(buf342, add_347, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_347
    del buf342
    del primals_265
    buf344 = buf343[0]
    buf345 = buf343[1]
    del buf343
    buf346 = empty((120, ), device='cpu', dtype=torch.float32)
    buf347 = empty((120, ), device='cpu', dtype=torch.float32)
    buf348 = buf315; del buf315  # reuse
    buf350 = buf348; del buf348  # reuse
    buf349 = buf347; del buf347  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_52(c_void_p(buf350.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(unsqueeze_722.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf346.data_ptr()))
    del convolution_71
    del primals_111
    del squeeze_166
    del unsqueeze_722
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf351 = aten.convolution_backward(buf350, mul_437, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_437
    del primals_264
    buf352 = buf351[0]
    buf353 = buf351[1]
    del buf351
    buf354 = reinterpret_tensor(buf328, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf328  # reuse
    buf355 = reinterpret_tensor(buf354, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf354  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_53(c_void_p(buf355.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(div_50.data_ptr()), c_void_p(bitwise_and_10.data_ptr()))
    del bitwise_and_10
    del div_50
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf356 = aten.convolution_backward(buf355, div_51, primals_262, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf355
    del div_51
    del primals_262
    buf357 = buf356[0]
    buf358 = buf356[1]
    buf359 = buf356[2]
    del buf356
    buf360 = reinterpret_tensor(buf357, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf357  # reuse
    cpp_fused_convolution_backward_hardswish_backward_54(c_void_p(buf360.data_ptr()), c_void_p(convolution_69.data_ptr()))
    del convolution_69
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf361 = aten.convolution_backward(buf360, mean_7, primals_260, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf360
    del mean_7
    del primals_260
    buf362 = buf361[0]
    buf363 = buf361[1]
    buf364 = buf361[2]
    del buf361
    buf365 = buf340; del buf340  # reuse
    buf366 = empty((360, ), device='cpu', dtype=torch.float32)
    buf367 = buf352; del buf352  # reuse
    buf368 = buf366; del buf366  # reuse
    buf369 = buf367; del buf367  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_55(c_void_p(buf369.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(clone_43.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(unsqueeze_734.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf365.data_ptr()))
    del clone_43
    del convolution_68
    del div_52
    del primals_109
    del squeeze_163
    del unsqueeze_734
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf370 = aten.convolution_backward(buf369, div_49, primals_259, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
    del buf369
    del div_49
    del primals_259
    buf371 = buf370[0]
    buf372 = buf370[1]
    del buf370
    buf373 = empty((360, ), device='cpu', dtype=torch.float32)
    buf374 = empty((360, ), device='cpu', dtype=torch.float32)
    buf375 = empty((360, ), device='cpu', dtype=torch.float32)
    buf376 = buf371; del buf371  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_56(c_void_p(buf376.data_ptr()), c_void_p(clone_42.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(unsqueeze_746.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    del clone_42
    del convolution_67
    del primals_107
    del squeeze_160
    del unsqueeze_746
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf377 = aten.convolution_backward(buf376, add_327, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_327
    del buf376
    del primals_258
    buf378 = buf377[0]
    buf379 = buf377[1]
    del buf377
    buf380 = buf245; del buf245  # reuse
    buf381 = empty((120, ), device='cpu', dtype=torch.float32)
    buf382 = empty((120, ), device='cpu', dtype=torch.float32)
    buf383 = empty((120, ), device='cpu', dtype=torch.float32)
    buf384 = buf350; del buf350  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_57(c_void_p(buf380.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(unsqueeze_758.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()))
    del buf278
    del buf311
    del buf344
    del buf378
    del convolution_66
    del primals_105
    del squeeze_157
    del unsqueeze_758
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf385 = aten.convolution_backward(buf384, mul_412, primals_257, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf384
    del mul_412
    del primals_257
    buf386 = buf385[0]
    buf387 = buf385[1]
    del buf385
    buf388 = reinterpret_tensor(buf362, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf362  # reuse
    buf389 = reinterpret_tensor(buf388, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf388  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_58(c_void_p(buf389.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(bitwise_and_11.data_ptr()))
    del bitwise_and_11
    del div_46
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf390 = aten.convolution_backward(buf389, div_47, primals_255, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf389
    del div_47
    del primals_255
    buf391 = buf390[0]
    buf392 = buf390[1]
    buf393 = buf390[2]
    del buf390
    buf394 = reinterpret_tensor(buf391, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf391  # reuse
    cpp_fused_convolution_backward_hardswish_backward_59(c_void_p(buf394.data_ptr()), c_void_p(convolution_64.data_ptr()))
    del convolution_64
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf395 = aten.convolution_backward(buf394, mean_6, primals_253, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf394
    del mean_6
    del primals_253
    buf396 = buf395[0]
    buf397 = buf395[1]
    buf398 = buf395[2]
    del buf395
    buf399 = buf374; del buf374  # reuse
    buf400 = empty((360, ), device='cpu', dtype=torch.float32)
    buf401 = buf386; del buf386  # reuse
    buf402 = buf400; del buf400  # reuse
    buf403 = buf401; del buf401  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_60(c_void_p(buf403.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(clone_40.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(unsqueeze_770.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf399.data_ptr()))
    del clone_40
    del convolution_63
    del div_48
    del primals_103
    del squeeze_154
    del unsqueeze_770
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf404 = aten.convolution_backward(buf403, div_45, primals_252, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False])
    del buf403
    del div_45
    del primals_252
    buf405 = buf404[0]
    buf406 = buf404[1]
    del buf404
    buf407 = empty((360, ), device='cpu', dtype=torch.float32)
    buf408 = empty((360, ), device='cpu', dtype=torch.float32)
    buf409 = empty((360, ), device='cpu', dtype=torch.float32)
    buf410 = buf405; del buf405  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_61(c_void_p(buf410.data_ptr()), c_void_p(clone_39.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(unsqueeze_782.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    del clone_39
    del convolution_62
    del primals_101
    del squeeze_151
    del unsqueeze_782
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf411 = aten.convolution_backward(buf410, add_307, primals_251, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_307
    del buf410
    del primals_251
    buf412 = buf411[0]
    buf413 = buf411[1]
    del buf411
    buf414 = buf382; del buf382  # reuse
    buf415 = empty((120, ), device='cpu', dtype=torch.float32)
    buf416 = empty((120, ), device='cpu', dtype=torch.float32)
    buf417 = buf380; del buf380  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_62(c_void_p(buf417.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_794.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()))
    del buf412
    del convolution_61
    del primals_99
    del squeeze_148
    del unsqueeze_794
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf418 = aten.convolution_backward(buf417, mul_387, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf417
    del mul_387
    del primals_250
    buf419 = buf418[0]
    buf420 = buf418[1]
    del buf418
    buf421 = reinterpret_tensor(buf396, (8, 360, 1, 1), (360, 1, 2880, 2880), 0); del buf396  # reuse
    buf422 = reinterpret_tensor(buf421, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf421  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_63(c_void_p(buf422.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(bitwise_and_12.data_ptr()))
    del bitwise_and_12
    del div_42
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf423 = aten.convolution_backward(buf422, div_43, primals_248, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf422
    del div_43
    del primals_248
    buf424 = buf423[0]
    buf425 = buf423[1]
    buf426 = buf423[2]
    del buf423
    buf427 = reinterpret_tensor(buf424, (8, 24, 1, 1), (24, 1, 1, 1), 0); del buf424  # reuse
    cpp_fused_convolution_backward_hardswish_backward_64(c_void_p(buf427.data_ptr()), c_void_p(convolution_59.data_ptr()))
    del convolution_59
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf428 = aten.convolution_backward(buf427, mean_5, primals_246, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf427
    del mean_5
    del primals_246
    buf429 = buf428[0]
    buf430 = buf428[1]
    buf431 = buf428[2]
    del buf428
    buf432 = buf408; del buf408  # reuse
    buf433 = empty((360, ), device='cpu', dtype=torch.float32)
    buf434 = buf419; del buf419  # reuse
    buf435 = buf433; del buf433  # reuse
    buf436 = buf434; del buf434  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_65(c_void_p(buf436.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(clone_37.data_ptr()), c_void_p(div_44.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(unsqueeze_806.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf432.data_ptr()))
    del buf429
    del clone_37
    del convolution_58
    del div_44
    del primals_97
    del squeeze_145
    del unsqueeze_806
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf437 = aten.convolution_backward(buf436, div_41, primals_245, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 360, [True, True, False])
    del buf436
    del div_41
    del primals_245
    buf438 = buf437[0]
    buf439 = buf437[1]
    del buf437
    buf440 = empty((360, ), device='cpu', dtype=torch.float32)
    buf441 = empty((360, ), device='cpu', dtype=torch.float32)
    buf442 = empty((360, ), device='cpu', dtype=torch.float32)
    buf443 = buf438; del buf438  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_66(c_void_p(buf443.data_ptr()), c_void_p(clone_36.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(unsqueeze_818.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()))
    del buf441
    del clone_36
    del convolution_57
    del primals_95
    del squeeze_142
    del unsqueeze_818
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf444 = aten.convolution_backward(buf443, add_288, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_288
    del buf443
    del primals_244
    buf445 = buf444[0]
    buf446 = buf444[1]
    del buf444
    buf447 = empty((72, ), device='cpu', dtype=torch.float32)
    buf448 = empty((72, ), device='cpu', dtype=torch.float32)
    buf449 = empty((72, ), device='cpu', dtype=torch.float32)
    buf450 = empty_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_67(c_void_p(buf445.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_830.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()))
    del convolution_56
    del primals_93
    del squeeze_139
    del unsqueeze_830
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf451 = aten.convolution_backward(buf450, div_40, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del div_40
    del primals_243
    buf452 = buf451[0]
    buf453 = buf451[1]
    del buf451
    buf454 = empty((216, ), device='cpu', dtype=torch.float32)
    buf455 = empty((216, ), device='cpu', dtype=torch.float32)
    buf456 = empty((216, ), device='cpu', dtype=torch.float32)
    buf457 = buf452; del buf452  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_68(c_void_p(buf457.data_ptr()), c_void_p(clone_35.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_842.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()))
    del clone_35
    del convolution_55
    del primals_91
    del squeeze_136
    del unsqueeze_842
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf458 = aten.convolution_backward(buf457, div_39, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False])
    del buf457
    del div_39
    del primals_242
    buf459 = buf458[0]
    buf460 = buf458[1]
    del buf458
    buf461 = buf455; del buf455  # reuse
    buf462 = empty((216, ), device='cpu', dtype=torch.float32)
    buf463 = empty((216, ), device='cpu', dtype=torch.float32)
    buf464 = buf459; del buf459  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_69(c_void_p(buf464.data_ptr()), c_void_p(clone_34.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_854.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()))
    del clone_34
    del convolution_54
    del primals_89
    del squeeze_133
    del unsqueeze_854
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf465 = aten.convolution_backward(buf464, add_270, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_270
    del buf464
    del primals_241
    buf466 = buf465[0]
    buf467 = buf465[1]
    del buf465
    buf468 = buf448; del buf448  # reuse
    buf469 = empty((72, ), device='cpu', dtype=torch.float32)
    buf470 = empty((72, ), device='cpu', dtype=torch.float32)
    buf471 = buf450; del buf450  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_70(c_void_p(buf445.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_866.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    del convolution_53
    del primals_87
    del squeeze_130
    del unsqueeze_866
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf472 = aten.convolution_backward(buf471, div_38, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del div_38
    del primals_240
    buf473 = buf472[0]
    buf474 = buf472[1]
    del buf472
    buf475 = buf462; del buf462  # reuse
    buf476 = empty((216, ), device='cpu', dtype=torch.float32)
    buf477 = empty((216, ), device='cpu', dtype=torch.float32)
    buf478 = buf473; del buf473  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_71(c_void_p(buf478.data_ptr()), c_void_p(clone_33.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_878.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    del clone_33
    del convolution_52
    del primals_85
    del squeeze_127
    del unsqueeze_878
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf479 = aten.convolution_backward(buf478, div_37, primals_239, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False])
    del buf478
    del div_37
    del primals_239
    buf480 = buf479[0]
    buf481 = buf479[1]
    del buf479
    buf482 = buf476; del buf476  # reuse
    buf483 = empty((216, ), device='cpu', dtype=torch.float32)
    buf484 = empty((216, ), device='cpu', dtype=torch.float32)
    buf485 = buf480; del buf480  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_72(c_void_p(buf485.data_ptr()), c_void_p(clone_32.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_890.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()))
    del clone_32
    del convolution_51
    del primals_83
    del squeeze_124
    del unsqueeze_890
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf486 = aten.convolution_backward(buf485, add_252, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_252
    del buf485
    del primals_238
    buf487 = buf486[0]
    buf488 = buf486[1]
    del buf486
    buf489 = buf469; del buf469  # reuse
    buf490 = empty((72, ), device='cpu', dtype=torch.float32)
    buf491 = buf471; del buf471  # reuse
    buf492 = buf490; del buf490  # reuse
    cpp_fused_add_native_batch_norm_backward_73(c_void_p(buf492.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_902.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf491.data_ptr()))
    del convolution_50
    del primals_81
    del squeeze_121
    del unsqueeze_902
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf493 = aten.convolution_backward(buf491, div_36, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del div_36
    del primals_237
    buf494 = buf493[0]
    buf495 = buf493[1]
    del buf493
    buf496 = buf483; del buf483  # reuse
    buf497 = empty((216, ), device='cpu', dtype=torch.float32)
    buf498 = empty((216, ), device='cpu', dtype=torch.float32)
    buf499 = buf494; del buf494  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_74(c_void_p(buf499.data_ptr()), c_void_p(clone_31.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_914.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()))
    del clone_31
    del convolution_49
    del primals_79
    del squeeze_118
    del unsqueeze_914
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf500 = aten.convolution_backward(buf499, div_35, primals_236, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False])
    del buf499
    del div_35
    del primals_236
    buf501 = buf500[0]
    buf502 = buf500[1]
    del buf500
    buf503 = buf497; del buf497  # reuse
    buf504 = empty((216, ), device='cpu', dtype=torch.float32)
    buf505 = empty((216, ), device='cpu', dtype=torch.float32)
    buf506 = buf501; del buf501  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_75(c_void_p(buf506.data_ptr()), c_void_p(clone_30.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_926.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()))
    del clone_30
    del convolution_48
    del primals_77
    del squeeze_115
    del unsqueeze_926
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf507 = aten.convolution_backward(buf506, add_234, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_234
    del buf506
    del primals_235
    buf508 = buf507[0]
    buf509 = buf507[1]
    del buf507
    buf510 = empty((72, ), device='cpu', dtype=torch.float32)
    buf511 = empty((72, ), device='cpu', dtype=torch.float32)
    buf512 = buf491; del buf491  # reuse
    buf514 = buf512; del buf512  # reuse
    buf513 = buf511; del buf511  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_76(c_void_p(buf514.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_938.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf510.data_ptr()))
    del convolution_47
    del primals_75
    del squeeze_112
    del unsqueeze_938
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf515 = aten.convolution_backward(buf514, div_34, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf514
    del div_34
    del primals_234
    buf516 = buf515[0]
    buf517 = buf515[1]
    del buf515
    buf518 = buf504; del buf504  # reuse
    buf519 = empty((216, ), device='cpu', dtype=torch.float32)
    buf520 = empty((216, ), device='cpu', dtype=torch.float32)
    buf521 = buf516; del buf516  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_77(c_void_p(buf521.data_ptr()), c_void_p(clone_29.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_950.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf520.data_ptr()))
    del clone_29
    del convolution_46
    del primals_73
    del squeeze_109
    del unsqueeze_950
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf522 = aten.convolution_backward(buf521, div_33, primals_233, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False])
    del buf521
    del div_33
    del primals_233
    buf523 = buf522[0]
    buf524 = buf522[1]
    del buf522
    buf525 = buf519; del buf519  # reuse
    buf526 = empty((216, ), device='cpu', dtype=torch.float32)
    buf527 = empty((216, ), device='cpu', dtype=torch.float32)
    buf528 = buf523; del buf523  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_78(c_void_p(buf528.data_ptr()), c_void_p(clone_28.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_962.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()))
    del buf526
    del clone_28
    del convolution_45
    del primals_71
    del squeeze_106
    del unsqueeze_962
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf529 = aten.convolution_backward(buf528, add_216, primals_232, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_216
    del buf528
    del primals_232
    buf530 = buf529[0]
    buf531 = buf529[1]
    del buf529
    buf532 = buf445; del buf445  # reuse
    buf533 = empty((72, ), device='cpu', dtype=torch.float32)
    buf534 = empty((72, ), device='cpu', dtype=torch.float32)
    buf535 = empty((72, ), device='cpu', dtype=torch.float32)
    buf536 = buf532; del buf532  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_79(c_void_p(buf536.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_974.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()))
    del buf466
    del buf487
    del buf508
    del buf530
    del buf534
    del convolution_44
    del primals_69
    del squeeze_103
    del unsqueeze_974
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf537 = aten.convolution_backward(buf536, div_32, primals_231, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf536
    del div_32
    del primals_231
    buf538 = buf537[0]
    buf539 = buf537[1]
    del buf537
    buf540 = empty((200, ), device='cpu', dtype=torch.float32)
    buf541 = empty((200, ), device='cpu', dtype=torch.float32)
    buf542 = empty((200, ), device='cpu', dtype=torch.float32)
    buf543 = buf538; del buf538  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_80(c_void_p(buf543.data_ptr()), c_void_p(clone_27.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_986.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()))
    del clone_27
    del convolution_43
    del primals_67
    del squeeze_100
    del unsqueeze_986
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf544 = aten.convolution_backward(buf543, div_31, primals_230, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 200, [True, True, False])
    del buf543
    del div_31
    del primals_230
    buf545 = buf544[0]
    buf546 = buf544[1]
    del buf544
    buf547 = buf541; del buf541  # reuse
    buf548 = empty((200, ), device='cpu', dtype=torch.float32)
    buf549 = empty((200, ), device='cpu', dtype=torch.float32)
    buf550 = buf545; del buf545  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_81(c_void_p(buf550.data_ptr()), c_void_p(clone_26.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_998.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()))
    del buf548
    del clone_26
    del convolution_42
    del primals_65
    del squeeze_97
    del unsqueeze_998
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf551 = aten.convolution_backward(buf550, add_199, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_199
    del buf550
    del primals_229
    buf552 = buf551[0]
    buf553 = buf551[1]
    del buf551
    buf554 = empty((40, ), device='cpu', dtype=torch.float32)
    buf555 = empty((40, ), device='cpu', dtype=torch.float32)
    buf556 = empty((40, ), device='cpu', dtype=torch.float32)
    buf557 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_82(c_void_p(buf552.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_1010.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()))
    del convolution_41
    del primals_63
    del squeeze_94
    del unsqueeze_1010
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf558 = aten.convolution_backward(buf557, mul_247, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_247
    del primals_228
    buf559 = buf558[0]
    buf560 = buf558[1]
    del buf558
    buf561 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf562 = reinterpret_tensor(buf561, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf561  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_83(c_void_p(buf562.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(bitwise_and_13.data_ptr()))
    del bitwise_and_13
    del div_28
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf563 = aten.convolution_backward(buf562, div_29, primals_226, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf562
    del div_29
    del primals_226
    buf564 = buf563[0]
    buf565 = buf563[1]
    buf566 = buf563[2]
    del buf563
    buf567 = reinterpret_tensor(buf564, (8, 16, 1, 1), (16, 1, 1, 1), 0); del buf564  # reuse
    cpp_fused_convolution_backward_hardswish_backward_84(c_void_p(buf567.data_ptr()), c_void_p(convolution_39.data_ptr()))
    del convolution_39
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf568 = aten.convolution_backward(buf567, mean_4, primals_224, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf567
    del mean_4
    del primals_224
    buf569 = buf568[0]
    buf570 = buf568[1]
    buf571 = buf568[2]
    del buf568
    buf572 = buf415; del buf415  # reuse
    buf573 = empty((120, ), device='cpu', dtype=torch.float32)
    buf574 = buf559; del buf559  # reuse
    buf575 = buf573; del buf573  # reuse
    buf576 = buf574; del buf574  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_85(c_void_p(buf576.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(clone_24.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_1022.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf572.data_ptr()))
    del clone_24
    del convolution_38
    del div_30
    del primals_61
    del squeeze_91
    del unsqueeze_1022
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf577 = aten.convolution_backward(buf576, div_27, primals_223, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf576
    del div_27
    del primals_223
    buf578 = buf577[0]
    buf579 = buf577[1]
    del buf577
    buf580 = empty((120, ), device='cpu', dtype=torch.float32)
    buf581 = empty((120, ), device='cpu', dtype=torch.float32)
    buf582 = empty((120, ), device='cpu', dtype=torch.float32)
    buf583 = buf578; del buf578  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_86(c_void_p(buf583.data_ptr()), c_void_p(clone_23.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_1034.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()))
    del clone_23
    del convolution_37
    del primals_59
    del squeeze_88
    del unsqueeze_1034
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf584 = aten.convolution_backward(buf583, add_179, primals_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_179
    del buf583
    del primals_222
    buf585 = buf584[0]
    buf586 = buf584[1]
    del buf584
    buf587 = buf555; del buf555  # reuse
    buf588 = empty((40, ), device='cpu', dtype=torch.float32)
    buf589 = empty((40, ), device='cpu', dtype=torch.float32)
    buf590 = buf557; del buf557  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_87(c_void_p(buf552.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_1046.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf590.data_ptr()))
    del convolution_36
    del primals_57
    del squeeze_85
    del unsqueeze_1046
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf591 = aten.convolution_backward(buf590, mul_222, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_222
    del primals_221
    buf592 = buf591[0]
    buf593 = buf591[1]
    del buf591
    buf594 = reinterpret_tensor(buf569, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf569  # reuse
    buf595 = reinterpret_tensor(buf594, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf594  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_88(c_void_p(buf595.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(bitwise_and_14.data_ptr()))
    del bitwise_and_14
    del div_24
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf596 = aten.convolution_backward(buf595, div_25, primals_219, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf595
    del div_25
    del primals_219
    buf597 = buf596[0]
    buf598 = buf596[1]
    buf599 = buf596[2]
    del buf596
    buf600 = reinterpret_tensor(buf597, (8, 16, 1, 1), (16, 1, 1, 1), 0); del buf597  # reuse
    cpp_fused_convolution_backward_hardswish_backward_89(c_void_p(buf600.data_ptr()), c_void_p(convolution_34.data_ptr()))
    del convolution_34
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf601 = aten.convolution_backward(buf600, mean_3, primals_217, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf600
    del mean_3
    del primals_217
    buf602 = buf601[0]
    buf603 = buf601[1]
    buf604 = buf601[2]
    del buf601
    buf605 = buf581; del buf581  # reuse
    buf606 = empty((120, ), device='cpu', dtype=torch.float32)
    buf607 = buf592; del buf592  # reuse
    buf608 = buf606; del buf606  # reuse
    buf609 = buf607; del buf607  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_90(c_void_p(buf609.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(clone_21.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_1058.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf605.data_ptr()))
    del clone_21
    del convolution_33
    del div_26
    del primals_55
    del squeeze_82
    del unsqueeze_1058
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf610 = aten.convolution_backward(buf609, div_23, primals_216, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf609
    del div_23
    del primals_216
    buf611 = buf610[0]
    buf612 = buf610[1]
    del buf610
    buf613 = empty((120, ), device='cpu', dtype=torch.float32)
    buf614 = empty((120, ), device='cpu', dtype=torch.float32)
    buf615 = empty((120, ), device='cpu', dtype=torch.float32)
    buf616 = buf611; del buf611  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_91(c_void_p(buf616.data_ptr()), c_void_p(clone_20.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_1070.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()))
    del clone_20
    del convolution_32
    del primals_53
    del squeeze_79
    del unsqueeze_1070
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf617 = aten.convolution_backward(buf616, add_159, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_159
    del buf616
    del primals_215
    buf618 = buf617[0]
    buf619 = buf617[1]
    del buf617
    buf620 = buf588; del buf588  # reuse
    buf621 = empty((40, ), device='cpu', dtype=torch.float32)
    buf622 = buf590; del buf590  # reuse
    buf623 = buf621; del buf621  # reuse
    cpp_fused_add_native_batch_norm_backward_92(c_void_p(buf623.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_1082.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf622.data_ptr()))
    del convolution_31
    del primals_51
    del squeeze_76
    del unsqueeze_1082
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf624 = aten.convolution_backward(buf622, mul_197, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_197
    del primals_214
    buf625 = buf624[0]
    buf626 = buf624[1]
    del buf624
    buf627 = reinterpret_tensor(buf602, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf602  # reuse
    buf628 = reinterpret_tensor(buf627, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf627  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_93(c_void_p(buf628.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(bitwise_and_15.data_ptr()))
    del bitwise_and_15
    del div_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf629 = aten.convolution_backward(buf628, div_21, primals_212, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf628
    del div_21
    del primals_212
    buf630 = buf629[0]
    buf631 = buf629[1]
    buf632 = buf629[2]
    del buf629
    buf633 = reinterpret_tensor(buf630, (8, 16, 1, 1), (16, 1, 1, 1), 0); del buf630  # reuse
    cpp_fused_convolution_backward_hardswish_backward_94(c_void_p(buf633.data_ptr()), c_void_p(convolution_29.data_ptr()))
    del convolution_29
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf634 = aten.convolution_backward(buf633, mean_2, primals_210, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf633
    del mean_2
    del primals_210
    buf635 = buf634[0]
    buf636 = buf634[1]
    buf637 = buf634[2]
    del buf634
    buf638 = buf614; del buf614  # reuse
    buf639 = empty((120, ), device='cpu', dtype=torch.float32)
    buf640 = buf625; del buf625  # reuse
    buf641 = buf639; del buf639  # reuse
    buf642 = buf640; del buf640  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_95(c_void_p(buf642.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(clone_18.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_1094.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf638.data_ptr()))
    del clone_18
    del convolution_28
    del div_22
    del primals_49
    del squeeze_73
    del unsqueeze_1094
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf643 = aten.convolution_backward(buf642, div_19, primals_209, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf642
    del div_19
    del primals_209
    buf644 = buf643[0]
    buf645 = buf643[1]
    del buf643
    buf646 = empty((120, ), device='cpu', dtype=torch.float32)
    buf647 = empty((120, ), device='cpu', dtype=torch.float32)
    buf648 = empty((120, ), device='cpu', dtype=torch.float32)
    buf649 = buf644; del buf644  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_96(c_void_p(buf649.data_ptr()), c_void_p(clone_17.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_1106.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()))
    del clone_17
    del convolution_27
    del primals_47
    del squeeze_70
    del unsqueeze_1106
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf650 = aten.convolution_backward(buf649, add_139, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_139
    del buf649
    del primals_208
    buf651 = buf650[0]
    buf652 = buf650[1]
    del buf650
    buf653 = empty((40, ), device='cpu', dtype=torch.float32)
    buf654 = empty((40, ), device='cpu', dtype=torch.float32)
    buf655 = buf622; del buf622  # reuse
    buf657 = buf655; del buf655  # reuse
    buf656 = buf654; del buf654  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_97(c_void_p(buf657.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_1118.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf653.data_ptr()))
    del convolution_26
    del primals_45
    del squeeze_67
    del unsqueeze_1118
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf658 = aten.convolution_backward(buf657, mul_172, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf657
    del mul_172
    del primals_207
    buf659 = buf658[0]
    buf660 = buf658[1]
    del buf658
    buf661 = reinterpret_tensor(buf635, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf635  # reuse
    buf662 = reinterpret_tensor(buf661, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf661  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_98(c_void_p(buf662.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(bitwise_and_16.data_ptr()))
    del bitwise_and_16
    del div_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf663 = aten.convolution_backward(buf662, div_17, primals_205, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf662
    del div_17
    del primals_205
    buf664 = buf663[0]
    buf665 = buf663[1]
    buf666 = buf663[2]
    del buf663
    buf667 = reinterpret_tensor(buf664, (8, 16, 1, 1), (16, 1, 1, 1), 0); del buf664  # reuse
    cpp_fused_convolution_backward_hardswish_backward_99(c_void_p(buf667.data_ptr()), c_void_p(convolution_24.data_ptr()))
    del convolution_24
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf668 = aten.convolution_backward(buf667, mean_1, primals_203, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf667
    del mean_1
    del primals_203
    buf669 = buf668[0]
    buf670 = buf668[1]
    buf671 = buf668[2]
    del buf668
    buf672 = buf647; del buf647  # reuse
    buf673 = empty((120, ), device='cpu', dtype=torch.float32)
    buf674 = buf659; del buf659  # reuse
    buf675 = buf673; del buf673  # reuse
    buf676 = buf674; del buf674  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_100(c_void_p(buf676.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(clone_15.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_1130.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf672.data_ptr()))
    del clone_15
    del convolution_23
    del div_18
    del primals_43
    del squeeze_64
    del unsqueeze_1130
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf677 = aten.convolution_backward(buf676, div_15, primals_202, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf676
    del div_15
    del primals_202
    buf678 = buf677[0]
    buf679 = buf677[1]
    del buf677
    buf680 = empty((120, ), device='cpu', dtype=torch.float32)
    buf681 = empty((120, ), device='cpu', dtype=torch.float32)
    buf682 = empty((120, ), device='cpu', dtype=torch.float32)
    buf683 = buf678; del buf678  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_101(c_void_p(buf683.data_ptr()), c_void_p(clone_14.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_1142.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()))
    del clone_14
    del convolution_22
    del primals_41
    del squeeze_61
    del unsqueeze_1142
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf684 = aten.convolution_backward(buf683, add_119, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_119
    del buf683
    del primals_201
    buf685 = buf684[0]
    buf686 = buf684[1]
    del buf684
    buf687 = buf552; del buf552  # reuse
    buf688 = empty((40, ), device='cpu', dtype=torch.float32)
    buf689 = empty((40, ), device='cpu', dtype=torch.float32)
    buf690 = empty((40, ), device='cpu', dtype=torch.float32)
    buf691 = buf687; del buf687  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_102(c_void_p(buf691.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_1154.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf690.data_ptr()))
    del buf585
    del buf618
    del buf651
    del buf685
    del buf689
    del convolution_21
    del primals_39
    del squeeze_58
    del unsqueeze_1154
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf692 = aten.convolution_backward(buf691, mul_147, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf691
    del mul_147
    del primals_200
    buf693 = buf692[0]
    buf694 = buf692[1]
    del buf692
    buf695 = reinterpret_tensor(buf669, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf669  # reuse
    buf696 = reinterpret_tensor(buf695, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf695  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_103(c_void_p(buf696.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(bitwise_and_17.data_ptr()))
    del bitwise_and_17
    del div_12
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf697 = aten.convolution_backward(buf696, div_13, primals_198, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf696
    del div_13
    del primals_198
    buf698 = buf697[0]
    buf699 = buf697[1]
    buf700 = buf697[2]
    del buf697
    buf701 = reinterpret_tensor(buf698, (8, 8, 1, 1), (8, 1, 1, 1), 0); del buf698  # reuse
    cpp_fused_convolution_backward_hardswish_backward_104(c_void_p(buf701.data_ptr()), c_void_p(convolution_19.data_ptr()))
    del convolution_19
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf702 = aten.convolution_backward(buf701, mean, primals_196, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean
    del primals_196
    buf703 = buf702[0]
    buf704 = buf702[1]
    buf705 = buf702[2]
    del buf702
    buf706 = buf681; del buf681  # reuse
    buf707 = empty((120, ), device='cpu', dtype=torch.float32)
    buf708 = buf693; del buf693  # reuse
    buf709 = buf707; del buf707  # reuse
    buf710 = buf708; del buf708  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_105(c_void_p(buf710.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(clone_12.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf703.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_1166.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf706.data_ptr()))
    del buf703
    del clone_12
    del convolution_18
    del div_14
    del primals_37
    del squeeze_55
    del unsqueeze_1166
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf711 = aten.convolution_backward(buf710, div_11, primals_195, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf710
    del div_11
    del primals_195
    buf712 = buf711[0]
    buf713 = buf711[1]
    del buf711
    buf714 = empty((120, ), device='cpu', dtype=torch.float32)
    buf715 = empty((120, ), device='cpu', dtype=torch.float32)
    buf716 = empty((120, ), device='cpu', dtype=torch.float32)
    buf717 = buf712; del buf712  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_106(c_void_p(buf717.data_ptr()), c_void_p(clone_11.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_1178.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()))
    del buf715
    del clone_11
    del convolution_17
    del primals_35
    del squeeze_52
    del unsqueeze_1178
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf718 = aten.convolution_backward(buf717, add_100, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_100
    del buf717
    del primals_194
    buf719 = buf718[0]
    buf720 = buf718[1]
    del buf718
    buf721 = empty((24, ), device='cpu', dtype=torch.float32)
    buf722 = empty((24, ), device='cpu', dtype=torch.float32)
    buf723 = empty((24, ), device='cpu', dtype=torch.float32)
    buf724 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_107(c_void_p(buf719.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_1190.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(buf724.data_ptr()))
    del convolution_16
    del primals_33
    del squeeze_49
    del unsqueeze_1190
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf725 = aten.convolution_backward(buf724, div_10, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del div_10
    del primals_193
    buf726 = buf725[0]
    buf727 = buf725[1]
    del buf725
    buf728 = empty((48, ), device='cpu', dtype=torch.float32)
    buf729 = empty((48, ), device='cpu', dtype=torch.float32)
    buf730 = empty((48, ), device='cpu', dtype=torch.float32)
    buf731 = buf726; del buf726  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_108(c_void_p(buf731.data_ptr()), c_void_p(clone_10.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_1202.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf730.data_ptr()))
    del clone_10
    del convolution_15
    del primals_31
    del squeeze_46
    del unsqueeze_1202
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf732 = aten.convolution_backward(buf731, div_9, primals_192, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False])
    del buf731
    del div_9
    del primals_192
    buf733 = buf732[0]
    buf734 = buf732[1]
    del buf732
    buf735 = buf729; del buf729  # reuse
    buf736 = empty((48, ), device='cpu', dtype=torch.float32)
    buf737 = empty((48, ), device='cpu', dtype=torch.float32)
    buf738 = buf733; del buf733  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_109(c_void_p(buf738.data_ptr()), c_void_p(clone_9.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_1214.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf735.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf737.data_ptr()))
    del clone_9
    del convolution_14
    del primals_29
    del squeeze_43
    del unsqueeze_1214
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf739 = aten.convolution_backward(buf738, add_82, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_82
    del buf738
    del primals_191
    buf740 = buf739[0]
    buf741 = buf739[1]
    del buf739
    buf742 = buf722; del buf722  # reuse
    buf743 = empty((24, ), device='cpu', dtype=torch.float32)
    buf744 = empty((24, ), device='cpu', dtype=torch.float32)
    buf745 = buf724; del buf724  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_110(c_void_p(buf719.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_1226.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf745.data_ptr()))
    del convolution_13
    del primals_27
    del squeeze_40
    del unsqueeze_1226
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf746 = aten.convolution_backward(buf745, div_8, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del div_8
    del primals_190
    buf747 = buf746[0]
    buf748 = buf746[1]
    del buf746
    buf749 = buf736; del buf736  # reuse
    buf750 = empty((48, ), device='cpu', dtype=torch.float32)
    buf751 = empty((48, ), device='cpu', dtype=torch.float32)
    buf752 = buf747; del buf747  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_111(c_void_p(buf752.data_ptr()), c_void_p(clone_8.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_1238.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf749.data_ptr()), c_void_p(buf750.data_ptr()), c_void_p(buf751.data_ptr()))
    del clone_8
    del convolution_12
    del primals_25
    del squeeze_37
    del unsqueeze_1238
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf753 = aten.convolution_backward(buf752, div_7, primals_189, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False])
    del buf752
    del div_7
    del primals_189
    buf754 = buf753[0]
    buf755 = buf753[1]
    del buf753
    buf756 = buf750; del buf750  # reuse
    buf757 = empty((48, ), device='cpu', dtype=torch.float32)
    buf758 = empty((48, ), device='cpu', dtype=torch.float32)
    buf759 = buf754; del buf754  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_112(c_void_p(buf759.data_ptr()), c_void_p(clone_7.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_1250.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf756.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf758.data_ptr()))
    del clone_7
    del convolution_11
    del primals_23
    del squeeze_34
    del unsqueeze_1250
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf760 = aten.convolution_backward(buf759, add_64, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_64
    del buf759
    del primals_188
    buf761 = buf760[0]
    buf762 = buf760[1]
    del buf760
    buf763 = buf743; del buf743  # reuse
    buf764 = empty((24, ), device='cpu', dtype=torch.float32)
    buf765 = buf745; del buf745  # reuse
    buf766 = buf764; del buf764  # reuse
    cpp_fused_add_native_batch_norm_backward_113(c_void_p(buf766.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_1262.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf765.data_ptr()))
    del convolution_10
    del primals_21
    del squeeze_31
    del unsqueeze_1262
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf767 = aten.convolution_backward(buf765, div_6, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf765
    del div_6
    del primals_187
    buf768 = buf767[0]
    buf769 = buf767[1]
    del buf767
    buf770 = buf757; del buf757  # reuse
    buf771 = empty((48, ), device='cpu', dtype=torch.float32)
    buf772 = empty((48, ), device='cpu', dtype=torch.float32)
    buf773 = buf768; del buf768  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_114(c_void_p(buf773.data_ptr()), c_void_p(clone_6.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_1274.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf771.data_ptr()), c_void_p(buf772.data_ptr()))
    del clone_6
    del convolution_9
    del primals_19
    del squeeze_28
    del unsqueeze_1274
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf774 = aten.convolution_backward(buf773, div_5, primals_186, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False])
    del buf773
    del div_5
    del primals_186
    buf775 = buf774[0]
    buf776 = buf774[1]
    del buf774
    buf777 = buf771; del buf771  # reuse
    buf778 = empty((48, ), device='cpu', dtype=torch.float32)
    buf779 = empty((48, ), device='cpu', dtype=torch.float32)
    buf780 = buf775; del buf775  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_115(c_void_p(buf780.data_ptr()), c_void_p(clone_5.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_1286.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(buf778.data_ptr()), c_void_p(buf779.data_ptr()))
    del buf778
    del clone_5
    del convolution_8
    del primals_17
    del squeeze_25
    del unsqueeze_1286
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf781 = aten.convolution_backward(buf780, add_46, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_46
    del buf780
    del primals_185
    buf782 = buf781[0]
    buf783 = buf781[1]
    del buf781
    buf784 = empty((24, ), device='cpu', dtype=torch.float32)
    buf785 = empty((24, ), device='cpu', dtype=torch.float32)
    buf786 = buf719; del buf719  # reuse
    buf788 = buf786; del buf786  # reuse
    buf787 = buf785; del buf785  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_116(c_void_p(buf788.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_1298.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf784.data_ptr()))
    del buf740
    del buf761
    del buf782
    del convolution_7
    del primals_15
    del squeeze_22
    del unsqueeze_1298
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf789 = aten.convolution_backward(buf788, div_4, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf788
    del div_4
    del primals_184
    buf790 = buf789[0]
    buf791 = buf789[1]
    del buf789
    buf792 = reinterpret_tensor(buf701, (64, ), (1, ), 0); del buf701  # reuse
    buf793 = empty((64, ), device='cpu', dtype=torch.float32)
    buf794 = empty((64, ), device='cpu', dtype=torch.float32)
    buf795 = buf790; del buf790  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_117(c_void_p(buf795.data_ptr()), c_void_p(clone_4.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_1310.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf794.data_ptr()))
    del clone_4
    del convolution_6
    del primals_13
    del squeeze_19
    del unsqueeze_1310
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf796 = aten.convolution_backward(buf795, div_3, primals_183, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 64, [True, True, False])
    del div_3
    del primals_183
    buf797 = buf796[0]
    buf798 = buf796[1]
    del buf796
    buf799 = buf793; del buf793  # reuse
    buf800 = empty((64, ), device='cpu', dtype=torch.float32)
    buf801 = empty((64, ), device='cpu', dtype=torch.float32)
    buf802 = buf797; del buf797  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_118(c_void_p(buf802.data_ptr()), c_void_p(clone_3.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_1322.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf801.data_ptr()))
    del buf800
    del clone_3
    del convolution_5
    del primals_11
    del squeeze_16
    del unsqueeze_1322
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf803 = aten.convolution_backward(buf802, add_29, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_29
    del buf802
    del primals_182
    buf804 = buf803[0]
    buf805 = buf803[1]
    del buf803
    buf806 = empty((16, ), device='cpu', dtype=torch.float32)
    buf807 = empty((16, ), device='cpu', dtype=torch.float32)
    buf808 = empty((16, ), device='cpu', dtype=torch.float32)
    buf809 = reinterpret_tensor(buf795, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf795  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_119(c_void_p(buf804.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_1334.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf809.data_ptr()))
    del convolution_4
    del primals_9
    del squeeze_13
    del unsqueeze_1334
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf810 = aten.convolution_backward(buf809, div_2, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf809
    del div_2
    del primals_181
    buf811 = buf810[0]
    buf812 = buf810[1]
    del buf810
    buf813 = buf807; del buf807  # reuse
    buf814 = empty((16, ), device='cpu', dtype=torch.float32)
    buf815 = empty((16, ), device='cpu', dtype=torch.float32)
    buf816 = buf811; del buf811  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_120(c_void_p(buf816.data_ptr()), c_void_p(clone_2.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_1346.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf815.data_ptr()))
    del clone_2
    del convolution_3
    del primals_7
    del squeeze_10
    del unsqueeze_1346
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf817 = aten.convolution_backward(buf816, add_17, primals_180, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
    del add_17
    del primals_180
    buf818 = buf817[0]
    buf819 = buf817[1]
    del buf817
    buf820 = buf814; del buf814  # reuse
    buf821 = empty((16, ), device='cpu', dtype=torch.float32)
    buf822 = empty((16, ), device='cpu', dtype=torch.float32)
    buf823 = buf816; del buf816  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_121(c_void_p(buf804.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_1358.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf820.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf822.data_ptr()), c_void_p(buf823.data_ptr()))
    del convolution_2
    del primals_5
    del squeeze_7
    del unsqueeze_1358
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf824 = aten.convolution_backward(buf823, div_1, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf823
    del div_1
    del primals_179
    buf825 = buf824[0]
    buf826 = buf824[1]
    del buf824
    buf827 = buf821; del buf821  # reuse
    buf828 = empty((16, ), device='cpu', dtype=torch.float32)
    buf829 = empty((16, ), device='cpu', dtype=torch.float32)
    buf830 = buf825; del buf825  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_122(c_void_p(buf830.data_ptr()), c_void_p(clone_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_1370.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf828.data_ptr()), c_void_p(buf829.data_ptr()))
    del clone_1
    del convolution_1
    del primals_3
    del squeeze_4
    del unsqueeze_1370
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf831 = aten.convolution_backward(buf830, div, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
    del buf830
    del div
    del primals_178
    buf832 = buf831[0]
    buf833 = buf831[1]
    del buf831
    buf834 = buf828; del buf828  # reuse
    buf835 = empty((16, ), device='cpu', dtype=torch.float32)
    buf836 = buf804; del buf804  # reuse
    buf838 = buf836; del buf836  # reuse
    buf837 = buf835; del buf835  # reuse
    cpp_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_123(c_void_p(buf838.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(clone.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(buf832.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_1382.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf834.data_ptr()))
    del buf818
    del buf832
    del clone
    del convolution
    del primals_1
    del squeeze_1
    del unsqueeze_1382
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf839 = aten.convolution_backward(buf838, primals_598, primals_177, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf838
    del primals_177
    del primals_598
    buf840 = buf839[1]
    return (buf837, buf834, buf829, buf827, buf822, buf820, buf815, buf813, buf808, buf806, buf801, buf799, buf794, buf792, buf787, buf784, buf779, buf777, buf772, buf770, buf766, buf763, buf758, buf756, buf751, buf749, buf744, buf742, buf737, buf735, buf730, buf728, buf723, buf721, buf716, buf714, buf709, buf706, buf690, buf688, buf682, buf680, buf675, buf672, buf656, buf653, buf648, buf646, buf641, buf638, buf623, buf620, buf615, buf613, buf608, buf605, buf589, buf587, buf582, buf580, buf575, buf572, buf556, buf554, buf549, buf547, buf542, buf540, buf535, buf533, buf527, buf525, buf520, buf518, buf513, buf510, buf505, buf503, buf498, buf496, buf492, buf489, buf484, buf482, buf477, buf475, buf470, buf468, buf463, buf461, buf456, buf454, buf449, buf447, buf442, buf440, buf435, buf432, buf416, buf414, buf409, buf407, buf402, buf399, buf383, buf381, buf375, buf373, buf368, buf365, buf349, buf346, buf341, buf339, buf334, buf331, buf316, buf313, buf308, buf306, buf301, buf298, buf282, buf280, buf275, buf273, buf268, buf265, buf249, buf247, buf242, buf240, buf235, buf232, buf216, buf214, buf209, buf207, buf202, buf199, buf183, buf181, buf175, buf173, buf168, buf165, buf149, buf146, buf141, buf139, buf134, buf131, buf116, buf113, buf108, buf106, buf101, buf98, buf82, buf80, buf75, buf73, buf68, buf65, buf49, buf47, buf42, buf40, buf35, buf32, buf16, buf14, buf9, buf7, reinterpret_tensor(buf1, (1000, 1984), (1984, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), buf840, buf833, buf826, buf819, buf812, buf805, buf798, buf791, buf783, buf776, buf769, buf762, buf755, buf748, buf741, buf734, buf727, buf720, buf713, buf704, buf705, buf699, buf700, buf694, buf686, buf679, buf670, buf671, buf665, buf666, buf660, buf652, buf645, buf636, buf637, buf631, buf632, buf626, buf619, buf612, buf603, buf604, buf598, buf599, buf593, buf586, buf579, buf570, buf571, buf565, buf566, buf560, buf553, buf546, buf539, buf531, buf524, buf517, buf509, buf502, buf495, buf488, buf481, buf474, buf467, buf460, buf453, buf446, buf439, buf430, buf431, buf425, buf426, buf420, buf413, buf406, buf397, buf398, buf392, buf393, buf387, buf379, buf372, buf363, buf364, buf358, buf359, buf353, buf345, buf338, buf329, buf330, buf324, buf325, buf319, buf312, buf305, buf296, buf297, buf291, buf292, buf286, buf279, buf272, buf263, buf264, buf258, buf259, buf253, buf246, buf239, buf230, buf231, buf225, buf226, buf220, buf213, buf206, buf197, buf198, buf192, buf193, buf187, buf179, buf172, buf163, buf164, buf158, buf159, buf153, buf145, buf138, buf129, buf130, buf124, buf125, buf119, buf112, buf105, buf96, buf97, buf91, buf92, buf86, buf79, buf72, buf63, buf64, buf58, buf59, buf53, buf46, buf39, buf30, buf31, buf25, buf26, buf20, buf13, buf6, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((120, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((8, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((120, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((200, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((200, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((72, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((360, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((360, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((24, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((360, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((720, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((720, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((32, 720, 1, 1), (720, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((720, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((184, 720, 1, 1), (720, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((48, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((1104, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((224, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((1984, 1344, 1, 1), (1344, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_598 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    clone_1 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    add_17 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    clone_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    add_29 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    clone_3 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    clone_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_46 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    clone_5 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    clone_6 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_64 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    clone_7 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    clone_8 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_82 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    clone_9 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    clone_10 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_100 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_11 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_12 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_147 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_119 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_14 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_15 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_172 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_139 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_17 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_18 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_197 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_159 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_20 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_21 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_222 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_179 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_23 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    clone_24 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    div_29 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_247 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_199 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 200, 28, 28), (156800, 1, 5600, 200), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    clone_26 = rand_strided((8, 200, 28, 28), (156800, 1, 5600, 200), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((8, 200, 28, 28), (156800, 1, 5600, 200), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    clone_27 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    add_216 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    clone_28 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    clone_29 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    add_234 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    clone_30 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    div_35 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    clone_31 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    add_252 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    clone_32 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    clone_33 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    div_38 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    add_270 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    clone_34 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((216, ), (1, ), device='cpu', dtype=torch.float32)
    clone_35 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    add_288 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_36 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_41 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_37 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((8, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    div_44 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    mul_387 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    add_307 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_39 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_40 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_47 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    mul_412 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    add_327 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_42 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_43 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_50 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    mul_437 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    add_347 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_45 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_53 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_46 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_56 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    mul_462 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    add_367 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_48 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_78 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_49 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_59 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    mul_487 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_81 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    add_387 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    convolution_82 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_187 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_51 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_83 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    squeeze_190 = rand_strided((360, ), (1, ), device='cpu', dtype=torch.float32)
    clone_52 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    div_62 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    convolution_84 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_63 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_64 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.float32)
    mul_512 = rand_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cpu', dtype=torch.float32)
    convolution_86 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    squeeze_193 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    add_407 = rand_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    convolution_87 = rand_strided((8, 720, 14, 14), (141120, 1, 10080, 720), device='cpu', dtype=torch.float32)
    squeeze_196 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    clone_54 = rand_strided((8, 720, 14, 14), (141120, 1, 10080, 720), device='cpu', dtype=torch.float32)
    div_65 = rand_strided((8, 720, 14, 14), (141120, 1, 10080, 720), device='cpu', dtype=torch.float32)
    convolution_88 = rand_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cpu', dtype=torch.float32)
    squeeze_199 = rand_strided((720, ), (1, ), device='cpu', dtype=torch.float32)
    clone_55 = rand_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cpu', dtype=torch.float32)
    div_66 = rand_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((8, 720, 1, 1), (720, 1, 720, 720), device='cpu', dtype=torch.float32)
    convolution_89 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_67 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_68 = rand_strided((8, 720, 1, 1), (720, 1, 720, 720), device='cpu', dtype=torch.float32)
    mul_537 = rand_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cpu', dtype=torch.float32)
    convolution_91 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    squeeze_202 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    add_426 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    convolution_92 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_205 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_57 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_69 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_93 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_208 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_58 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_70 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    mean_12 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    convolution_94 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_71 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_72 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    mul_562 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_96 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    squeeze_211 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    add_446 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    convolution_97 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_214 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_60 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_73 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_98 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_217 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_61 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_74 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    mean_13 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    convolution_99 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_75 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_76 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    mul_587 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_101 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    squeeze_220 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    add_466 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    convolution_102 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_223 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_63 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_77 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_103 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_226 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_64 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_78 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    mean_14 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    convolution_104 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_79 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_80 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    mul_612 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_106 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    squeeze_229 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    add_486 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    convolution_107 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_232 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_66 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_81 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_108 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_235 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_67 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_82 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    mean_15 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    convolution_109 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_83 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_84 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    mul_637 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_111 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    squeeze_238 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    add_506 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    convolution_112 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_241 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_69 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_85 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_113 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    squeeze_244 = rand_strided((736, ), (1, ), device='cpu', dtype=torch.float32)
    clone_70 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    div_86 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    mean_16 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    convolution_114 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_87 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_88 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.float32)
    mul_662 = rand_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cpu', dtype=torch.float32)
    convolution_116 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    squeeze_247 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    add_526 = rand_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cpu', dtype=torch.float32)
    convolution_117 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cpu', dtype=torch.float32)
    squeeze_250 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    clone_72 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cpu', dtype=torch.float32)
    div_89 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cpu', dtype=torch.float32)
    convolution_118 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cpu', dtype=torch.float32)
    squeeze_253 = rand_strided((1104, ), (1, ), device='cpu', dtype=torch.float32)
    clone_73 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cpu', dtype=torch.float32)
    div_90 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cpu', dtype=torch.float32)
    mean_17 = rand_strided((8, 1104, 1, 1), (1104, 1, 1104, 1104), device='cpu', dtype=torch.float32)
    convolution_119 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_91 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    div_92 = rand_strided((8, 1104, 1, 1), (1104, 1, 1104, 1104), device='cpu', dtype=torch.float32)
    mul_687 = rand_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cpu', dtype=torch.float32)
    convolution_121 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    squeeze_256 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    add_545 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_122 = rand_strided((8, 1344, 7, 7), (65856, 1, 9408, 1344), device='cpu', dtype=torch.float32)
    squeeze_259 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    clone_75 = rand_strided((8, 1344, 7, 7), (65856, 1, 9408, 1344), device='cpu', dtype=torch.float32)
    mean_18 = rand_strided((8, 1344, 1, 1), (1344, 1, 1344, 1344), device='cpu', dtype=torch.float32)
    convolution_123 = rand_strided((8, 1984, 1, 1), (1984, 1, 1984, 1984), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((8, 1984), (1984, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1984), (1984, 1), device='cpu', dtype=torch.float32)
    unsqueeze_350 = rand_strided((1, 1344, 1, 1), (1344, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and = rand_strided((8, 1104, 1, 1), (1104, 1, 1104, 1104), device='cpu', dtype=torch.bool)
    unsqueeze_374 = rand_strided((1, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 1104, 1, 1), (1104, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.bool)
    unsqueeze_410 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_2 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.bool)
    unsqueeze_446 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_470 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_3 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.bool)
    unsqueeze_482 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_494 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_4 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.bool)
    unsqueeze_518 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_530 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_542 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_5 = rand_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cpu', dtype=torch.bool)
    unsqueeze_554 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_566 = rand_strided((1, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_578 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_6 = rand_strided((8, 720, 1, 1), (720, 1, 720, 720), device='cpu', dtype=torch.bool)
    unsqueeze_590 = rand_strided((1, 720, 1, 1), (720, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_602 = rand_strided((1, 720, 1, 1), (720, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_614 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_7 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.bool)
    unsqueeze_626 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_638 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_650 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_8 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.bool)
    unsqueeze_662 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_674 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_686 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_9 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.bool)
    unsqueeze_698 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_710 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_722 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_10 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.bool)
    unsqueeze_734 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_11 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.bool)
    unsqueeze_770 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_782 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_794 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_12 = rand_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cpu', dtype=torch.bool)
    unsqueeze_806 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_818 = rand_strided((1, 360, 1, 1), (360, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_830 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_842 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_854 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_866 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_878 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_890 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_902 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_914 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_926 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_938 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_950 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_962 = rand_strided((1, 216, 1, 1), (216, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_974 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_986 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_998 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1010 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_13 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    unsqueeze_1022 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1034 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1046 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_14 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    unsqueeze_1058 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1070 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1082 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_15 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    unsqueeze_1094 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1106 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1118 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_16 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    unsqueeze_1130 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1142 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1154 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_17 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    unsqueeze_1166 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1178 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1190 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1202 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1214 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1226 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1238 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1250 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1262 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1274 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1286 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1298 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1310 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1322 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1334 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1346 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1358 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1370 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1382 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_264, primals_265, primals_266, primals_267, primals_269, primals_271, primals_272, primals_273, primals_274, primals_276, primals_278, primals_279, primals_280, primals_281, primals_283, primals_285, primals_286, primals_287, primals_288, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_299, primals_300, primals_301, primals_302, primals_304, primals_306, primals_307, primals_308, primals_309, primals_311, primals_313, primals_314, primals_315, primals_316, primals_318, primals_320, primals_321, primals_322, primals_323, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_334, primals_335, primals_336, primals_598, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, clone_1, div_1, convolution_2, squeeze_7, add_17, convolution_3, squeeze_10, clone_2, div_2, convolution_4, squeeze_13, add_29, convolution_5, squeeze_16, clone_3, div_3, convolution_6, squeeze_19, clone_4, div_4, convolution_7, squeeze_22, add_46, convolution_8, squeeze_25, clone_5, div_5, convolution_9, squeeze_28, clone_6, div_6, convolution_10, squeeze_31, add_64, convolution_11, squeeze_34, clone_7, div_7, convolution_12, squeeze_37, clone_8, div_8, convolution_13, squeeze_40, add_82, convolution_14, squeeze_43, clone_9, div_9, convolution_15, squeeze_46, clone_10, div_10, convolution_16, squeeze_49, add_100, convolution_17, squeeze_52, clone_11, div_11, convolution_18, squeeze_55, clone_12, div_12, mean, convolution_19, div_13, div_14, mul_147, convolution_21, squeeze_58, add_119, convolution_22, squeeze_61, clone_14, div_15, convolution_23, squeeze_64, clone_15, div_16, mean_1, convolution_24, div_17, div_18, mul_172, convolution_26, squeeze_67, add_139, convolution_27, squeeze_70, clone_17, div_19, convolution_28, squeeze_73, clone_18, div_20, mean_2, convolution_29, div_21, div_22, mul_197, convolution_31, squeeze_76, add_159, convolution_32, squeeze_79, clone_20, div_23, convolution_33, squeeze_82, clone_21, div_24, mean_3, convolution_34, div_25, div_26, mul_222, convolution_36, squeeze_85, add_179, convolution_37, squeeze_88, clone_23, div_27, convolution_38, squeeze_91, clone_24, div_28, mean_4, convolution_39, div_29, div_30, mul_247, convolution_41, squeeze_94, add_199, convolution_42, squeeze_97, clone_26, div_31, convolution_43, squeeze_100, clone_27, div_32, convolution_44, squeeze_103, add_216, convolution_45, squeeze_106, clone_28, div_33, convolution_46, squeeze_109, clone_29, div_34, convolution_47, squeeze_112, add_234, convolution_48, squeeze_115, clone_30, div_35, convolution_49, squeeze_118, clone_31, div_36, convolution_50, squeeze_121, add_252, convolution_51, squeeze_124, clone_32, div_37, convolution_52, squeeze_127, clone_33, div_38, convolution_53, squeeze_130, add_270, convolution_54, squeeze_133, clone_34, div_39, convolution_55, squeeze_136, clone_35, div_40, convolution_56, squeeze_139, add_288, convolution_57, squeeze_142, clone_36, div_41, convolution_58, squeeze_145, clone_37, div_42, mean_5, convolution_59, div_43, div_44, mul_387, convolution_61, squeeze_148, add_307, convolution_62, squeeze_151, clone_39, div_45, convolution_63, squeeze_154, clone_40, div_46, mean_6, convolution_64, div_47, div_48, mul_412, convolution_66, squeeze_157, add_327, convolution_67, squeeze_160, clone_42, div_49, convolution_68, squeeze_163, clone_43, div_50, mean_7, convolution_69, div_51, div_52, mul_437, convolution_71, squeeze_166, add_347, convolution_72, squeeze_169, clone_45, div_53, convolution_73, squeeze_172, clone_46, div_54, mean_8, convolution_74, div_55, div_56, mul_462, convolution_76, squeeze_175, add_367, convolution_77, squeeze_178, clone_48, div_57, convolution_78, squeeze_181, clone_49, div_58, mean_9, convolution_79, div_59, div_60, mul_487, convolution_81, squeeze_184, add_387, convolution_82, squeeze_187, clone_51, div_61, convolution_83, squeeze_190, clone_52, div_62, mean_10, convolution_84, div_63, div_64, mul_512, convolution_86, squeeze_193, add_407, convolution_87, squeeze_196, clone_54, div_65, convolution_88, squeeze_199, clone_55, div_66, mean_11, convolution_89, div_67, div_68, mul_537, convolution_91, squeeze_202, add_426, convolution_92, squeeze_205, clone_57, div_69, convolution_93, squeeze_208, clone_58, div_70, mean_12, convolution_94, div_71, div_72, mul_562, convolution_96, squeeze_211, add_446, convolution_97, squeeze_214, clone_60, div_73, convolution_98, squeeze_217, clone_61, div_74, mean_13, convolution_99, div_75, div_76, mul_587, convolution_101, squeeze_220, add_466, convolution_102, squeeze_223, clone_63, div_77, convolution_103, squeeze_226, clone_64, div_78, mean_14, convolution_104, div_79, div_80, mul_612, convolution_106, squeeze_229, add_486, convolution_107, squeeze_232, clone_66, div_81, convolution_108, squeeze_235, clone_67, div_82, mean_15, convolution_109, div_83, div_84, mul_637, convolution_111, squeeze_238, add_506, convolution_112, squeeze_241, clone_69, div_85, convolution_113, squeeze_244, clone_70, div_86, mean_16, convolution_114, div_87, div_88, mul_662, convolution_116, squeeze_247, add_526, convolution_117, squeeze_250, clone_72, div_89, convolution_118, squeeze_253, clone_73, div_90, mean_17, convolution_119, div_91, div_92, mul_687, convolution_121, squeeze_256, add_545, convolution_122, squeeze_259, clone_75, mean_18, convolution_123, view_1, permute_1, unsqueeze_350, unsqueeze_362, bitwise_and, unsqueeze_374, unsqueeze_386, unsqueeze_398, bitwise_and_1, unsqueeze_410, unsqueeze_422, unsqueeze_434, bitwise_and_2, unsqueeze_446, unsqueeze_458, unsqueeze_470, bitwise_and_3, unsqueeze_482, unsqueeze_494, unsqueeze_506, bitwise_and_4, unsqueeze_518, unsqueeze_530, unsqueeze_542, bitwise_and_5, unsqueeze_554, unsqueeze_566, unsqueeze_578, bitwise_and_6, unsqueeze_590, unsqueeze_602, unsqueeze_614, bitwise_and_7, unsqueeze_626, unsqueeze_638, unsqueeze_650, bitwise_and_8, unsqueeze_662, unsqueeze_674, unsqueeze_686, bitwise_and_9, unsqueeze_698, unsqueeze_710, unsqueeze_722, bitwise_and_10, unsqueeze_734, unsqueeze_746, unsqueeze_758, bitwise_and_11, unsqueeze_770, unsqueeze_782, unsqueeze_794, bitwise_and_12, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, bitwise_and_13, unsqueeze_1022, unsqueeze_1034, unsqueeze_1046, bitwise_and_14, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, bitwise_and_15, unsqueeze_1094, unsqueeze_1106, unsqueeze_1118, bitwise_and_16, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, bitwise_and_17, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, unsqueeze_1298, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1382, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetv3_b', benchmark_compiled_module)
