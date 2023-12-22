
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(10240L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
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
                        tmp37.store(out_ptr3 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_2 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
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
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
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
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (160L*x1)));
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (160L*x0)));
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
                tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
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
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (672L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (112L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (112L*x0)));
                }
            }
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
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
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (672L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (112L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (112L*x0)));
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                }
            }
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


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
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
                        tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (184L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (184L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
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
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (184L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (184L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (80L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (80L*x0)));
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
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_34 = async_compile.cpp('''
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


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_35 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (80L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (80L*x0)));
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (240L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (240L*x0)));
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
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_39 = async_compile.cpp('''
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


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_40 = async_compile.cpp('''
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


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_42 = async_compile.cpp('''
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp17 = static_cast<float>(0.00015943877551020407);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = tmp20 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        auto tmp23 = tmp15 * tmp22;
                        auto tmp24 = tmp12 - tmp23;
                        auto tmp26 = tmp25 * tmp18;
                        auto tmp27 = tmp24 - tmp26;
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
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


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.cpp('''
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
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00015943877551020407);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_44 = async_compile.cpp('''
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


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_45 = async_compile.cpp('''
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


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_47 = async_compile.cpp('''
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp17 = static_cast<float>(0.00015943877551020407);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = tmp20 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        auto tmp23 = tmp15 * tmp22;
                        auto tmp24 = tmp12 - tmp23;
                        auto tmp26 = tmp25 * tmp18;
                        auto tmp27 = tmp24 - tmp26;
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
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


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_48 = async_compile.cpp('''
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
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00015943877551020407);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_50 = async_compile.cpp('''
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(72L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp17 = static_cast<float>(0.00015943877551020407);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = tmp20 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        auto tmp23 = tmp15 * tmp22;
                        auto tmp24 = tmp12 - tmp23;
                        auto tmp26 = tmp25 * tmp18;
                        auto tmp27 = tmp24 - tmp26;
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                    }
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (72L*x0)));
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
                    auto tmp10 = static_cast<float>(3.985969387755102e-05);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_54 = async_compile.cpp('''
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


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (72L*x0)));
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
                    auto tmp10 = static_cast<float>(3.985969387755102e-05);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (72L*x0)));
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
                    auto tmp10 = static_cast<float>(3.985969387755102e-05);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_57 = async_compile.cpp('''
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.cpp('''
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
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    auto tmp10 = static_cast<float>(3.985969387755102e-05);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_59 = async_compile.cpp('''
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
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    auto tmp10 = static_cast<float>(9.964923469387754e-06);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_60 = async_compile.cpp('''
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


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_61 = async_compile.cpp('''
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
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
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
                    auto tmp10 = static_cast<float>(9.964923469387754e-06);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_hardswish_backward_native_batch_norm_backward_62 = async_compile.cpp('''
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
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
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
                        auto tmp21 = tmp19 - tmp20;
                        auto tmp22 = tmp18 * tmp21;
                        tmp_acc0_vec = tmp_acc0_vec + tmp18;
                        tmp_acc1_vec = tmp_acc1_vec + tmp22;
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = static_cast<float>(9.964923469387754e-06);
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp27 = tmp26 * tmp26;
                    auto tmp28 = tmp25 * tmp27;
                    auto tmp29 = tmp21 * tmp28;
                    auto tmp30 = tmp18 - tmp29;
                    auto tmp32 = tmp31 * tmp24;
                    auto tmp33 = tmp30 - tmp32;
                    auto tmp35 = tmp26 * tmp34;
                    auto tmp36 = tmp33 * tmp35;
                    tmp36.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_164, primals_165, primals_166, primals_167, primals_169, primals_171, primals_172, primals_173, primals_313, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, relu, convolution_2, squeeze_7, add_16, convolution_3, squeeze_10, relu_1, convolution_4, squeeze_13, relu_2, convolution_5, squeeze_16, add_31, convolution_6, squeeze_19, relu_3, convolution_7, squeeze_22, relu_4, convolution_8, squeeze_25, add_47, convolution_9, squeeze_28, relu_5, convolution_10, squeeze_31, relu_6, mean, relu_7, div_1, mul_78, convolution_13, squeeze_34, add_63, convolution_14, squeeze_37, relu_8, convolution_15, squeeze_40, relu_9, mean_1, relu_10, div_2, mul_100, convolution_18, squeeze_43, add_80, convolution_19, squeeze_46, relu_11, convolution_20, squeeze_49, relu_12, mean_2, relu_13, div_3, mul_122, convolution_23, squeeze_52, add_97, convolution_24, squeeze_55, clone_1, div_4, convolution_25, squeeze_58, clone_2, div_5, convolution_26, squeeze_61, add_114, convolution_27, squeeze_64, clone_3, div_6, convolution_28, squeeze_67, clone_4, div_7, convolution_29, squeeze_70, add_132, convolution_30, squeeze_73, clone_5, div_8, convolution_31, squeeze_76, clone_6, div_9, convolution_32, squeeze_79, add_150, convolution_33, squeeze_82, clone_7, div_10, convolution_34, squeeze_85, clone_8, div_11, convolution_35, squeeze_88, add_168, convolution_36, squeeze_91, clone_9, div_12, convolution_37, squeeze_94, clone_10, div_13, mean_3, relu_14, div_14, mul_238, convolution_40, squeeze_97, add_186, convolution_41, squeeze_100, clone_11, div_15, convolution_42, squeeze_103, clone_12, div_16, mean_4, relu_15, div_17, mul_262, convolution_45, squeeze_106, add_205, convolution_46, squeeze_109, clone_13, div_18, convolution_47, squeeze_112, clone_14, div_19, mean_5, relu_16, div_20, mul_286, convolution_50, squeeze_115, add_223, convolution_51, squeeze_118, clone_15, div_21, convolution_52, squeeze_121, clone_16, div_22, mean_6, relu_17, div_23, mul_310, convolution_55, squeeze_124, add_242, convolution_56, squeeze_127, clone_17, div_24, convolution_57, squeeze_130, clone_18, div_25, mean_7, relu_18, div_26, mul_334, convolution_60, squeeze_133, add_261, convolution_61, squeeze_136, clone_19, mean_8, convolution_62, view_1, permute_1, unsqueeze_186, unsqueeze_198, bitwise_and, unsqueeze_210, unsqueeze_222, unsqueeze_234, bitwise_and_1, unsqueeze_246, unsqueeze_258, unsqueeze_270, bitwise_and_2, unsqueeze_282, unsqueeze_294, unsqueeze_306, bitwise_and_3, unsqueeze_318, unsqueeze_330, unsqueeze_342, bitwise_and_4, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, unsqueeze_498, unsqueeze_510, unsqueeze_522, bitwise_and_5, unsqueeze_534, unsqueeze_546, unsqueeze_558, bitwise_and_6, unsqueeze_570, unsqueeze_582, unsqueeze_594, bitwise_and_7, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_13, (72, ), (1, ))
    assert_size_stride(primals_15, (72, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (72, ), (1, ))
    assert_size_stride(primals_21, (72, ), (1, ))
    assert_size_stride(primals_23, (40, ), (1, ))
    assert_size_stride(primals_25, (120, ), (1, ))
    assert_size_stride(primals_27, (120, ), (1, ))
    assert_size_stride(primals_29, (40, ), (1, ))
    assert_size_stride(primals_31, (120, ), (1, ))
    assert_size_stride(primals_33, (120, ), (1, ))
    assert_size_stride(primals_35, (40, ), (1, ))
    assert_size_stride(primals_37, (240, ), (1, ))
    assert_size_stride(primals_39, (240, ), (1, ))
    assert_size_stride(primals_41, (80, ), (1, ))
    assert_size_stride(primals_43, (200, ), (1, ))
    assert_size_stride(primals_45, (200, ), (1, ))
    assert_size_stride(primals_47, (80, ), (1, ))
    assert_size_stride(primals_49, (184, ), (1, ))
    assert_size_stride(primals_51, (184, ), (1, ))
    assert_size_stride(primals_53, (80, ), (1, ))
    assert_size_stride(primals_55, (184, ), (1, ))
    assert_size_stride(primals_57, (184, ), (1, ))
    assert_size_stride(primals_59, (80, ), (1, ))
    assert_size_stride(primals_61, (480, ), (1, ))
    assert_size_stride(primals_63, (480, ), (1, ))
    assert_size_stride(primals_65, (112, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_69, (672, ), (1, ))
    assert_size_stride(primals_71, (112, ), (1, ))
    assert_size_stride(primals_73, (672, ), (1, ))
    assert_size_stride(primals_75, (672, ), (1, ))
    assert_size_stride(primals_77, (160, ), (1, ))
    assert_size_stride(primals_79, (960, ), (1, ))
    assert_size_stride(primals_81, (960, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_85, (960, ), (1, ))
    assert_size_stride(primals_87, (960, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_91, (960, ), (1, ))
    assert_size_stride(primals_95, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_96, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_97, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_98, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_99, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_100, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_101, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_102, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_103, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_104, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_105, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_106, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_108, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_110, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_111, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_112, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_113, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_115, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_117, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_118, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_119, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_120, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_122, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_124, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_125, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_126, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_127, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_128, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_129, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_130, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_131, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_132, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_134, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_135, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_136, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_137, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_138, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_139, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_141, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_143, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_144, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_145, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_148, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_150, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_151, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_152, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_153, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_155, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_157, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_158, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_159, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_160, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_162, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_164, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_165, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_166, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_167, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_169, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_171, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_172, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_173, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_313, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(clone, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_1, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_4, (16, ), (1, ))
    assert_size_stride(relu, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_16, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_10, (64, ), (1, ))
    assert_size_stride(relu_1, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(relu_2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_5, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_16, (24, ), (1, ))
    assert_size_stride(add_31, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_6, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(squeeze_19, (72, ), (1, ))
    assert_size_stride(relu_3, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_7, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(squeeze_22, (72, ), (1, ))
    assert_size_stride(relu_4, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_8, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_25, (24, ), (1, ))
    assert_size_stride(add_47, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_9, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(squeeze_28, (72, ), (1, ))
    assert_size_stride(relu_5, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_10, (8, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(squeeze_31, (72, ), (1, ))
    assert_size_stride(relu_6, (8, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(mean, (8, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(relu_7, (8, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(div_1, (8, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(mul_78, (8, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(convolution_13, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_34, (40, ), (1, ))
    assert_size_stride(add_63, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_14, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_37, (120, ), (1, ))
    assert_size_stride(relu_8, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_15, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_40, (120, ), (1, ))
    assert_size_stride(relu_9, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_1, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(relu_10, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_2, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_100, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_18, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_43, (40, ), (1, ))
    assert_size_stride(add_80, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_19, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_46, (120, ), (1, ))
    assert_size_stride(relu_11, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_20, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_49, (120, ), (1, ))
    assert_size_stride(relu_12, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_2, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(relu_13, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_3, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_122, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_23, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_52, (40, ), (1, ))
    assert_size_stride(add_97, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_24, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_55, (240, ), (1, ))
    assert_size_stride(clone_1, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(div_4, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_25, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(squeeze_58, (240, ), (1, ))
    assert_size_stride(clone_2, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(div_5, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_26, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_61, (80, ), (1, ))
    assert_size_stride(add_114, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_27, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(squeeze_64, (200, ), (1, ))
    assert_size_stride(clone_3, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_6, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_28, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(squeeze_67, (200, ), (1, ))
    assert_size_stride(clone_4, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_7, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_29, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_70, (80, ), (1, ))
    assert_size_stride(add_132, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_30, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(squeeze_73, (184, ), (1, ))
    assert_size_stride(clone_5, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_8, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_31, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(squeeze_76, (184, ), (1, ))
    assert_size_stride(clone_6, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_9, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_32, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_79, (80, ), (1, ))
    assert_size_stride(add_150, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_33, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(squeeze_82, (184, ), (1, ))
    assert_size_stride(clone_7, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_10, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_34, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(squeeze_85, (184, ), (1, ))
    assert_size_stride(clone_8, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_11, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_35, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_88, (80, ), (1, ))
    assert_size_stride(add_168, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_36, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_91, (480, ), (1, ))
    assert_size_stride(clone_9, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(div_12, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_37, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_94, (480, ), (1, ))
    assert_size_stride(clone_10, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(div_13, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_3, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(relu_14, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(div_14, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_238, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_40, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_97, (112, ), (1, ))
    assert_size_stride(add_186, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_41, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_100, (672, ), (1, ))
    assert_size_stride(clone_11, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_15, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_42, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_103, (672, ), (1, ))
    assert_size_stride(clone_12, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_16, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_4, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_15, (8, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_17, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_262, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_45, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_106, (112, ), (1, ))
    assert_size_stride(add_205, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_46, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_109, (672, ), (1, ))
    assert_size_stride(clone_13, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_18, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_47, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(squeeze_112, (672, ), (1, ))
    assert_size_stride(clone_14, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(div_19, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(mean_5, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_16, (8, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_20, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_286, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(convolution_50, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_115, (160, ), (1, ))
    assert_size_stride(add_223, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_51, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_118, (960, ), (1, ))
    assert_size_stride(clone_15, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_21, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_52, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_121, (960, ), (1, ))
    assert_size_stride(clone_16, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_22, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_6, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_17, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_23, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_310, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_55, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_124, (160, ), (1, ))
    assert_size_stride(add_242, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_56, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_127, (960, ), (1, ))
    assert_size_stride(clone_17, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_24, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_57, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_130, (960, ), (1, ))
    assert_size_stride(clone_18, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_25, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_7, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_18, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_26, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_334, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_60, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_133, (160, ), (1, ))
    assert_size_stride(add_261, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_61, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_136, (960, ), (1, ))
    assert_size_stride(clone_19, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_8, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(convolution_62, (8, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(view_1, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(unsqueeze_186, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_198, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_and, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(unsqueeze_210, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_222, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_234, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_and_1, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(unsqueeze_246, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_270, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_and_2, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(unsqueeze_282, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(bitwise_and_3, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(unsqueeze_318, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_342, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(bitwise_and_4, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(unsqueeze_354, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_486, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_522, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_5, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_534, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_558, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_6, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(unsqueeze_570, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_594, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(bitwise_and_7, (8, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(unsqueeze_606, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_630, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_666, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_702, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view_1, out=buf1)
    del view_1
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = reinterpret_tensor(buf0, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf0  # reuse
    cpp_fused_convolution_backward_hardswish_backward_sum_0(c_void_p(buf3.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(buf2.data_ptr()))
    del convolution_62
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf4 = aten.convolution_backward(buf3, mean_8, primals_173, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf3
    del mean_8
    del primals_173
    buf5 = buf4[0]
    buf6 = buf4[1]
    buf7 = buf4[2]
    del buf4
    buf8 = empty((960, ), device='cpu', dtype=torch.float32)
    buf9 = empty((960, ), device='cpu', dtype=torch.float32)
    buf10 = empty((960, ), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_1(c_void_p(clone_19.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_186.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    del clone_19
    del convolution_61
    del primals_91
    del squeeze_136
    del unsqueeze_186
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf12 = aten.convolution_backward(buf11, add_261, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_261
    del buf11
    del primals_172
    buf13 = buf12[0]
    buf14 = buf12[1]
    del buf12
    buf15 = empty((160, ), device='cpu', dtype=torch.float32)
    buf16 = empty((160, ), device='cpu', dtype=torch.float32)
    buf17 = empty((160, ), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_2(c_void_p(buf13.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_198.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del convolution_60
    del primals_89
    del squeeze_133
    del unsqueeze_198
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf19 = aten.convolution_backward(buf18, mul_334, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_334
    del primals_171
    buf20 = buf19[0]
    buf21 = buf19[1]
    del buf19
    buf22 = reinterpret_tensor(buf5, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf5  # reuse
    buf23 = reinterpret_tensor(buf22, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf22  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_3(c_void_p(buf23.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(bitwise_and.data_ptr()))
    del bitwise_and
    del div_25
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf24 = aten.convolution_backward(buf23, relu_18, primals_169, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf23
    del primals_169
    buf25 = buf24[0]
    buf26 = buf24[1]
    buf27 = buf24[2]
    del buf24
    buf28 = buf25; del buf25  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_4(c_void_p(buf28.data_ptr()), c_void_p(relu_18.data_ptr()))
    del relu_18
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf29 = aten.convolution_backward(buf28, mean_7, primals_167, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf28
    del mean_7
    del primals_167
    buf30 = buf29[0]
    buf31 = buf29[1]
    buf32 = buf29[2]
    del buf29
    buf33 = buf9; del buf9  # reuse
    buf34 = empty((960, ), device='cpu', dtype=torch.float32)
    buf35 = buf20; del buf20  # reuse
    buf36 = buf34; del buf34  # reuse
    buf37 = buf35; del buf35  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_5(c_void_p(buf37.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(clone_18.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(unsqueeze_210.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf33.data_ptr()))
    del clone_18
    del convolution_57
    del div_26
    del primals_87
    del squeeze_130
    del unsqueeze_210
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf38 = aten.convolution_backward(buf37, div_24, primals_166, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False])
    del buf37
    del div_24
    del primals_166
    buf39 = buf38[0]
    buf40 = buf38[1]
    del buf38
    buf41 = empty((960, ), device='cpu', dtype=torch.float32)
    buf42 = empty((960, ), device='cpu', dtype=torch.float32)
    buf43 = empty((960, ), device='cpu', dtype=torch.float32)
    buf44 = buf39; del buf39  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_6(c_void_p(buf44.data_ptr()), c_void_p(clone_17.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_222.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del clone_17
    del convolution_56
    del primals_85
    del squeeze_127
    del unsqueeze_222
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf45 = aten.convolution_backward(buf44, add_242, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_242
    del buf44
    del primals_165
    buf46 = buf45[0]
    buf47 = buf45[1]
    del buf45
    buf48 = buf16; del buf16  # reuse
    buf49 = empty((160, ), device='cpu', dtype=torch.float32)
    buf50 = empty((160, ), device='cpu', dtype=torch.float32)
    buf51 = buf18; del buf18  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_7(c_void_p(buf13.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del convolution_55
    del primals_83
    del squeeze_124
    del unsqueeze_234
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf52 = aten.convolution_backward(buf51, mul_310, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf51
    del mul_310
    del primals_164
    buf53 = buf52[0]
    buf54 = buf52[1]
    del buf52
    buf55 = reinterpret_tensor(buf30, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf30  # reuse
    buf56 = reinterpret_tensor(buf55, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf55  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_8(c_void_p(buf56.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(bitwise_and_1.data_ptr()))
    del bitwise_and_1
    del div_22
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf57 = aten.convolution_backward(buf56, relu_17, primals_162, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf56
    del primals_162
    buf58 = buf57[0]
    buf59 = buf57[1]
    buf60 = buf57[2]
    del buf57
    buf61 = buf58; del buf58  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_9(c_void_p(buf61.data_ptr()), c_void_p(relu_17.data_ptr()))
    del relu_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf62 = aten.convolution_backward(buf61, mean_6, primals_160, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf61
    del mean_6
    del primals_160
    buf63 = buf62[0]
    buf64 = buf62[1]
    buf65 = buf62[2]
    del buf62
    buf66 = buf42; del buf42  # reuse
    buf67 = empty((960, ), device='cpu', dtype=torch.float32)
    buf68 = buf53; del buf53  # reuse
    buf69 = buf67; del buf67  # reuse
    buf70 = buf68; del buf68  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_10(c_void_p(buf70.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(clone_16.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf66.data_ptr()))
    del buf63
    del clone_16
    del convolution_52
    del div_23
    del primals_81
    del squeeze_121
    del unsqueeze_246
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf71 = aten.convolution_backward(buf70, div_21, primals_159, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False])
    del buf70
    del div_21
    del primals_159
    buf72 = buf71[0]
    buf73 = buf71[1]
    del buf71
    buf74 = empty((960, ), device='cpu', dtype=torch.float32)
    buf75 = empty((960, ), device='cpu', dtype=torch.float32)
    buf76 = empty((960, ), device='cpu', dtype=torch.float32)
    buf77 = buf72; del buf72  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_11(c_void_p(buf77.data_ptr()), c_void_p(clone_15.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    del buf75
    del clone_15
    del convolution_51
    del primals_79
    del squeeze_118
    del unsqueeze_258
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf78 = aten.convolution_backward(buf77, add_223, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_223
    del buf77
    del primals_158
    buf79 = buf78[0]
    buf80 = buf78[1]
    del buf78
    buf81 = buf49; del buf49  # reuse
    buf82 = empty((160, ), device='cpu', dtype=torch.float32)
    buf83 = buf13; del buf13  # reuse
    buf84 = buf82; del buf82  # reuse
    cpp_fused_add_native_batch_norm_backward_12(c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_270.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf81.data_ptr()))
    del buf46
    del buf79
    del convolution_50
    del primals_77
    del squeeze_115
    del unsqueeze_270
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf85 = aten.convolution_backward(buf83, mul_286, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf83
    del mul_286
    del primals_157
    buf86 = buf85[0]
    buf87 = buf85[1]
    del buf85
    buf88 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf89 = reinterpret_tensor(buf88, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf88  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_13(c_void_p(buf89.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(bitwise_and_2.data_ptr()))
    del bitwise_and_2
    del div_19
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf90 = aten.convolution_backward(buf89, relu_16, primals_155, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf89
    del primals_155
    buf91 = buf90[0]
    buf92 = buf90[1]
    buf93 = buf90[2]
    del buf90
    buf94 = buf91; del buf91  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_14(c_void_p(buf94.data_ptr()), c_void_p(relu_16.data_ptr()))
    del relu_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf95 = aten.convolution_backward(buf94, mean_5, primals_153, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf94
    del mean_5
    del primals_153
    buf96 = buf95[0]
    buf97 = buf95[1]
    buf98 = buf95[2]
    del buf95
    buf99 = empty((672, ), device='cpu', dtype=torch.float32)
    buf100 = empty((672, ), device='cpu', dtype=torch.float32)
    buf101 = buf86; del buf86  # reuse
    buf102 = buf100; del buf100  # reuse
    buf103 = buf101; del buf101  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_15(c_void_p(buf103.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(clone_14.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf99.data_ptr()))
    del clone_14
    del convolution_47
    del div_20
    del primals_75
    del squeeze_112
    del unsqueeze_282
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf104 = aten.convolution_backward(buf103, div_18, primals_152, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf103
    del div_18
    del primals_152
    buf105 = buf104[0]
    buf106 = buf104[1]
    del buf104
    buf107 = empty((672, ), device='cpu', dtype=torch.float32)
    buf108 = empty((672, ), device='cpu', dtype=torch.float32)
    buf109 = empty((672, ), device='cpu', dtype=torch.float32)
    buf110 = buf105; del buf105  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_16(c_void_p(buf110.data_ptr()), c_void_p(clone_13.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_294.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del clone_13
    del convolution_46
    del primals_73
    del squeeze_109
    del unsqueeze_294
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf111 = aten.convolution_backward(buf110, add_205, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_205
    del buf110
    del primals_151
    buf112 = buf111[0]
    buf113 = buf111[1]
    del buf111
    buf114 = empty((112, ), device='cpu', dtype=torch.float32)
    buf115 = empty((112, ), device='cpu', dtype=torch.float32)
    buf116 = empty((112, ), device='cpu', dtype=torch.float32)
    buf117 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_17(c_void_p(buf112.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del convolution_45
    del primals_71
    del squeeze_106
    del unsqueeze_306
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf118 = aten.convolution_backward(buf117, mul_262, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf117
    del mul_262
    del primals_150
    buf119 = buf118[0]
    buf120 = buf118[1]
    del buf118
    buf121 = reinterpret_tensor(buf96, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf96  # reuse
    buf122 = reinterpret_tensor(buf121, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf121  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_18(c_void_p(buf122.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(bitwise_and_3.data_ptr()))
    del bitwise_and_3
    del div_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf123 = aten.convolution_backward(buf122, relu_15, primals_148, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf122
    del primals_148
    buf124 = buf123[0]
    buf125 = buf123[1]
    buf126 = buf123[2]
    del buf123
    buf127 = buf124; del buf124  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_19(c_void_p(buf127.data_ptr()), c_void_p(relu_15.data_ptr()))
    del relu_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf128 = aten.convolution_backward(buf127, mean_4, primals_146, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf127
    del mean_4
    del primals_146
    buf129 = buf128[0]
    buf130 = buf128[1]
    buf131 = buf128[2]
    del buf128
    buf132 = buf108; del buf108  # reuse
    buf133 = empty((672, ), device='cpu', dtype=torch.float32)
    buf134 = buf119; del buf119  # reuse
    buf135 = buf133; del buf133  # reuse
    buf136 = buf134; del buf134  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_20(c_void_p(buf136.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(clone_12.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_318.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf132.data_ptr()))
    del buf129
    del clone_12
    del convolution_42
    del div_17
    del primals_69
    del squeeze_103
    del unsqueeze_318
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf137 = aten.convolution_backward(buf136, div_15, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf136
    del div_15
    del primals_145
    buf138 = buf137[0]
    buf139 = buf137[1]
    del buf137
    buf140 = empty((672, ), device='cpu', dtype=torch.float32)
    buf141 = empty((672, ), device='cpu', dtype=torch.float32)
    buf142 = empty((672, ), device='cpu', dtype=torch.float32)
    buf143 = buf138; del buf138  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_21(c_void_p(buf143.data_ptr()), c_void_p(clone_11.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    del buf141
    del clone_11
    del convolution_41
    del primals_67
    del squeeze_100
    del unsqueeze_330
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf144 = aten.convolution_backward(buf143, add_186, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_186
    del buf143
    del primals_144
    buf145 = buf144[0]
    buf146 = buf144[1]
    del buf144
    buf147 = buf115; del buf115  # reuse
    buf148 = empty((112, ), device='cpu', dtype=torch.float32)
    buf149 = empty((112, ), device='cpu', dtype=torch.float32)
    buf150 = buf112; del buf112  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_22(c_void_p(buf150.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    del buf145
    del buf148
    del convolution_40
    del primals_65
    del squeeze_97
    del unsqueeze_342
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf151 = aten.convolution_backward(buf150, mul_238, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf150
    del mul_238
    del primals_143
    buf152 = buf151[0]
    buf153 = buf151[1]
    del buf151
    buf154 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf155 = reinterpret_tensor(buf154, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf154  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_23(c_void_p(buf155.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(bitwise_and_4.data_ptr()))
    del bitwise_and_4
    del div_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf156 = aten.convolution_backward(buf155, relu_14, primals_141, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf155
    del primals_141
    buf157 = buf156[0]
    buf158 = buf156[1]
    buf159 = buf156[2]
    del buf156
    buf160 = buf157; del buf157  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_24(c_void_p(buf160.data_ptr()), c_void_p(relu_14.data_ptr()))
    del relu_14
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf161 = aten.convolution_backward(buf160, mean_3, primals_139, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_3
    del primals_139
    buf162 = buf161[0]
    buf163 = buf161[1]
    buf164 = buf161[2]
    del buf161
    buf165 = empty((480, ), device='cpu', dtype=torch.float32)
    buf166 = empty((480, ), device='cpu', dtype=torch.float32)
    buf167 = buf152; del buf152  # reuse
    buf168 = buf166; del buf166  # reuse
    buf169 = buf167; del buf167  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_25(c_void_p(buf169.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(clone_10.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf165.data_ptr()))
    del buf162
    del clone_10
    del convolution_37
    del div_14
    del primals_63
    del squeeze_94
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf170 = aten.convolution_backward(buf169, div_12, primals_138, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf169
    del div_12
    del primals_138
    buf171 = buf170[0]
    buf172 = buf170[1]
    del buf170
    buf173 = empty((480, ), device='cpu', dtype=torch.float32)
    buf174 = empty((480, ), device='cpu', dtype=torch.float32)
    buf175 = empty((480, ), device='cpu', dtype=torch.float32)
    buf176 = buf171; del buf171  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_26(c_void_p(buf176.data_ptr()), c_void_p(clone_9.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    del buf174
    del clone_9
    del convolution_36
    del primals_61
    del squeeze_91
    del unsqueeze_366
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf177 = aten.convolution_backward(buf176, add_168, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_168
    del buf176
    del primals_137
    buf178 = buf177[0]
    buf179 = buf177[1]
    del buf177
    buf180 = empty((80, ), device='cpu', dtype=torch.float32)
    buf181 = empty((80, ), device='cpu', dtype=torch.float32)
    buf182 = empty((80, ), device='cpu', dtype=torch.float32)
    buf183 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_27(c_void_p(buf178.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    del convolution_35
    del primals_59
    del squeeze_88
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf184 = aten.convolution_backward(buf183, div_11, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del div_11
    del primals_136
    buf185 = buf184[0]
    buf186 = buf184[1]
    del buf184
    buf187 = empty((184, ), device='cpu', dtype=torch.float32)
    buf188 = empty((184, ), device='cpu', dtype=torch.float32)
    buf189 = empty((184, ), device='cpu', dtype=torch.float32)
    buf190 = buf185; del buf185  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_28(c_void_p(buf190.data_ptr()), c_void_p(clone_8.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    del clone_8
    del convolution_34
    del primals_57
    del squeeze_85
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf191 = aten.convolution_backward(buf190, div_10, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False])
    del buf190
    del div_10
    del primals_135
    buf192 = buf191[0]
    buf193 = buf191[1]
    del buf191
    buf194 = buf188; del buf188  # reuse
    buf195 = empty((184, ), device='cpu', dtype=torch.float32)
    buf196 = empty((184, ), device='cpu', dtype=torch.float32)
    buf197 = buf192; del buf192  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_29(c_void_p(buf197.data_ptr()), c_void_p(clone_7.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()))
    del clone_7
    del convolution_33
    del primals_55
    del squeeze_82
    del unsqueeze_402
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf198 = aten.convolution_backward(buf197, add_150, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_150
    del buf197
    del primals_134
    buf199 = buf198[0]
    buf200 = buf198[1]
    del buf198
    buf201 = buf181; del buf181  # reuse
    buf202 = empty((80, ), device='cpu', dtype=torch.float32)
    buf203 = empty((80, ), device='cpu', dtype=torch.float32)
    buf204 = buf183; del buf183  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_30(c_void_p(buf178.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    del convolution_32
    del primals_53
    del squeeze_79
    del unsqueeze_414
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf205 = aten.convolution_backward(buf204, div_9, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del div_9
    del primals_133
    buf206 = buf205[0]
    buf207 = buf205[1]
    del buf205
    buf208 = buf195; del buf195  # reuse
    buf209 = empty((184, ), device='cpu', dtype=torch.float32)
    buf210 = empty((184, ), device='cpu', dtype=torch.float32)
    buf211 = buf206; del buf206  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31(c_void_p(buf211.data_ptr()), c_void_p(clone_6.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    del clone_6
    del convolution_31
    del primals_51
    del squeeze_76
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf212 = aten.convolution_backward(buf211, div_8, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False])
    del buf211
    del div_8
    del primals_132
    buf213 = buf212[0]
    buf214 = buf212[1]
    del buf212
    buf215 = buf209; del buf209  # reuse
    buf216 = empty((184, ), device='cpu', dtype=torch.float32)
    buf217 = empty((184, ), device='cpu', dtype=torch.float32)
    buf218 = buf213; del buf213  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_32(c_void_p(buf218.data_ptr()), c_void_p(clone_5.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    del buf216
    del clone_5
    del convolution_30
    del primals_49
    del squeeze_73
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf219 = aten.convolution_backward(buf218, add_132, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_132
    del buf218
    del primals_131
    buf220 = buf219[0]
    buf221 = buf219[1]
    del buf219
    buf222 = buf202; del buf202  # reuse
    buf223 = empty((80, ), device='cpu', dtype=torch.float32)
    buf224 = buf204; del buf204  # reuse
    buf225 = buf223; del buf223  # reuse
    cpp_fused_add_native_batch_norm_backward_33(c_void_p(buf225.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()))
    del convolution_29
    del primals_47
    del squeeze_70
    del unsqueeze_450
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf226 = aten.convolution_backward(buf224, div_7, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf224
    del div_7
    del primals_130
    buf227 = buf226[0]
    buf228 = buf226[1]
    del buf226
    buf229 = empty((200, ), device='cpu', dtype=torch.float32)
    buf230 = empty((200, ), device='cpu', dtype=torch.float32)
    buf231 = empty((200, ), device='cpu', dtype=torch.float32)
    buf232 = buf227; del buf227  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_34(c_void_p(buf232.data_ptr()), c_void_p(clone_4.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del clone_4
    del convolution_28
    del primals_45
    del squeeze_67
    del unsqueeze_462
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf233 = aten.convolution_backward(buf232, div_6, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 200, [True, True, False])
    del buf232
    del div_6
    del primals_129
    buf234 = buf233[0]
    buf235 = buf233[1]
    del buf233
    buf236 = buf230; del buf230  # reuse
    buf237 = empty((200, ), device='cpu', dtype=torch.float32)
    buf238 = empty((200, ), device='cpu', dtype=torch.float32)
    buf239 = buf234; del buf234  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_35(c_void_p(buf239.data_ptr()), c_void_p(clone_3.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del buf237
    del clone_3
    del convolution_27
    del primals_43
    del squeeze_64
    del unsqueeze_474
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf240 = aten.convolution_backward(buf239, add_114, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_114
    del buf239
    del primals_128
    buf241 = buf240[0]
    buf242 = buf240[1]
    del buf240
    buf243 = empty((80, ), device='cpu', dtype=torch.float32)
    buf244 = empty((80, ), device='cpu', dtype=torch.float32)
    buf245 = buf178; del buf178  # reuse
    buf247 = buf245; del buf245  # reuse
    buf246 = buf244; del buf244  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_36(c_void_p(buf247.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf243.data_ptr()))
    del buf199
    del buf220
    del buf241
    del convolution_26
    del primals_41
    del squeeze_61
    del unsqueeze_486
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf248 = aten.convolution_backward(buf247, div_5, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf247
    del div_5
    del primals_127
    buf249 = buf248[0]
    buf250 = buf248[1]
    del buf248
    buf251 = empty((240, ), device='cpu', dtype=torch.float32)
    buf252 = empty((240, ), device='cpu', dtype=torch.float32)
    buf253 = empty((240, ), device='cpu', dtype=torch.float32)
    buf254 = buf249; del buf249  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_37(c_void_p(buf254.data_ptr()), c_void_p(clone_2.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del clone_2
    del convolution_25
    del primals_39
    del squeeze_58
    del unsqueeze_498
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf255 = aten.convolution_backward(buf254, div_4, primals_126, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf254
    del div_4
    del primals_126
    buf256 = buf255[0]
    buf257 = buf255[1]
    del buf255
    buf258 = buf252; del buf252  # reuse
    buf259 = empty((240, ), device='cpu', dtype=torch.float32)
    buf260 = empty((240, ), device='cpu', dtype=torch.float32)
    buf261 = buf256; del buf256  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_38(c_void_p(buf261.data_ptr()), c_void_p(clone_1.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del buf259
    del clone_1
    del convolution_24
    del primals_37
    del squeeze_55
    del unsqueeze_510
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf262 = aten.convolution_backward(buf261, add_97, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_97
    del buf261
    del primals_125
    buf263 = buf262[0]
    buf264 = buf262[1]
    del buf262
    buf265 = empty((40, ), device='cpu', dtype=torch.float32)
    buf266 = empty((40, ), device='cpu', dtype=torch.float32)
    buf267 = empty((40, ), device='cpu', dtype=torch.float32)
    buf268 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_39(c_void_p(buf263.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_522.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    del convolution_23
    del primals_35
    del squeeze_52
    del unsqueeze_522
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf269 = aten.convolution_backward(buf268, mul_122, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_122
    del primals_124
    buf270 = buf269[0]
    buf271 = buf269[1]
    del buf269
    buf272 = reinterpret_tensor(buf160, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf160  # reuse
    buf273 = reinterpret_tensor(buf272, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf272  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_40(c_void_p(buf273.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(bitwise_and_5.data_ptr()))
    del bitwise_and_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf274 = aten.convolution_backward(buf273, relu_13, primals_122, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf273
    del primals_122
    buf275 = buf274[0]
    buf276 = buf274[1]
    buf277 = buf274[2]
    del buf274
    buf278 = buf275; del buf275  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_41(c_void_p(buf278.data_ptr()), c_void_p(relu_13.data_ptr()))
    del relu_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf279 = aten.convolution_backward(buf278, mean_2, primals_120, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf278
    del mean_2
    del primals_120
    buf280 = buf279[0]
    buf281 = buf279[1]
    buf282 = buf279[2]
    del buf279
    buf283 = empty((120, ), device='cpu', dtype=torch.float32)
    buf284 = empty((120, ), device='cpu', dtype=torch.float32)
    buf285 = buf270; del buf270  # reuse
    buf286 = buf284; del buf284  # reuse
    buf287 = buf285; del buf285  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_42(c_void_p(buf287.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_534.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf283.data_ptr()))
    del convolution_20
    del div_3
    del primals_33
    del relu_12
    del squeeze_49
    del unsqueeze_534
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf288 = aten.convolution_backward(buf287, relu_11, primals_119, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf287
    del primals_119
    buf289 = buf288[0]
    buf290 = buf288[1]
    del buf288
    buf291 = empty((120, ), device='cpu', dtype=torch.float32)
    buf292 = empty((120, ), device='cpu', dtype=torch.float32)
    buf293 = empty((120, ), device='cpu', dtype=torch.float32)
    buf294 = buf289; del buf289  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(buf294.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    del convolution_19
    del primals_31
    del relu_11
    del squeeze_46
    del unsqueeze_546
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf295 = aten.convolution_backward(buf294, add_80, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_80
    del buf294
    del primals_118
    buf296 = buf295[0]
    buf297 = buf295[1]
    del buf295
    buf298 = buf266; del buf266  # reuse
    buf299 = empty((40, ), device='cpu', dtype=torch.float32)
    buf300 = empty((40, ), device='cpu', dtype=torch.float32)
    buf301 = buf268; del buf268  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_44(c_void_p(buf263.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_558.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    del convolution_18
    del primals_29
    del squeeze_43
    del unsqueeze_558
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf302 = aten.convolution_backward(buf301, mul_100, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf301
    del mul_100
    del primals_117
    buf303 = buf302[0]
    buf304 = buf302[1]
    del buf302
    buf305 = reinterpret_tensor(buf280, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf280  # reuse
    buf306 = reinterpret_tensor(buf305, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf305  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_45(c_void_p(buf306.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(bitwise_and_6.data_ptr()))
    del bitwise_and_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf307 = aten.convolution_backward(buf306, relu_10, primals_115, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf306
    del primals_115
    buf308 = buf307[0]
    buf309 = buf307[1]
    buf310 = buf307[2]
    del buf307
    buf311 = buf308; del buf308  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_46(c_void_p(buf311.data_ptr()), c_void_p(relu_10.data_ptr()))
    del relu_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf312 = aten.convolution_backward(buf311, mean_1, primals_113, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf311
    del mean_1
    del primals_113
    buf313 = buf312[0]
    buf314 = buf312[1]
    buf315 = buf312[2]
    del buf312
    buf316 = buf292; del buf292  # reuse
    buf317 = empty((120, ), device='cpu', dtype=torch.float32)
    buf318 = buf303; del buf303  # reuse
    buf319 = buf317; del buf317  # reuse
    buf320 = buf318; del buf318  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_47(c_void_p(buf320.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_570.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf316.data_ptr()))
    del buf313
    del convolution_15
    del div_2
    del primals_27
    del relu_9
    del squeeze_40
    del unsqueeze_570
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf321 = aten.convolution_backward(buf320, relu_8, primals_112, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf320
    del primals_112
    buf322 = buf321[0]
    buf323 = buf321[1]
    del buf321
    buf324 = empty((120, ), device='cpu', dtype=torch.float32)
    buf325 = empty((120, ), device='cpu', dtype=torch.float32)
    buf326 = empty((120, ), device='cpu', dtype=torch.float32)
    buf327 = buf322; del buf322  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_48(c_void_p(buf327.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del buf325
    del convolution_14
    del primals_25
    del relu_8
    del squeeze_37
    del unsqueeze_582
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf328 = aten.convolution_backward(buf327, add_63, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_63
    del buf327
    del primals_111
    buf329 = buf328[0]
    buf330 = buf328[1]
    del buf328
    buf331 = buf299; del buf299  # reuse
    buf332 = empty((40, ), device='cpu', dtype=torch.float32)
    buf333 = buf263; del buf263  # reuse
    buf334 = buf332; del buf332  # reuse
    cpp_fused_add_native_batch_norm_backward_49(c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_594.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf331.data_ptr()))
    del buf296
    del buf329
    del convolution_13
    del primals_23
    del squeeze_34
    del unsqueeze_594
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf335 = aten.convolution_backward(buf333, mul_78, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf333
    del mul_78
    del primals_110
    buf336 = buf335[0]
    buf337 = buf335[1]
    del buf335
    buf338 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cpu', dtype=torch.float32)
    buf339 = reinterpret_tensor(buf338, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf338  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_50(c_void_p(buf339.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(bitwise_and_7.data_ptr()))
    del bitwise_and_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf340 = aten.convolution_backward(buf339, relu_7, primals_108, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf339
    del primals_108
    buf341 = buf340[0]
    buf342 = buf340[1]
    buf343 = buf340[2]
    del buf340
    buf344 = buf341; del buf341  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_51(c_void_p(buf344.data_ptr()), c_void_p(relu_7.data_ptr()))
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf345 = aten.convolution_backward(buf344, mean, primals_106, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf344
    del mean
    del primals_106
    buf346 = buf345[0]
    buf347 = buf345[1]
    buf348 = buf345[2]
    del buf345
    buf349 = empty((72, ), device='cpu', dtype=torch.float32)
    buf350 = empty((72, ), device='cpu', dtype=torch.float32)
    buf351 = buf336; del buf336  # reuse
    buf352 = buf350; del buf350  # reuse
    buf353 = buf351; del buf351  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_52(c_void_p(buf353.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_606.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf349.data_ptr()))
    del buf346
    del convolution_10
    del div_1
    del primals_21
    del relu_6
    del squeeze_31
    del unsqueeze_606
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf354 = aten.convolution_backward(buf353, relu_5, primals_105, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False])
    del buf353
    del primals_105
    buf355 = buf354[0]
    buf356 = buf354[1]
    del buf354
    buf357 = empty((72, ), device='cpu', dtype=torch.float32)
    buf358 = empty((72, ), device='cpu', dtype=torch.float32)
    buf359 = empty((72, ), device='cpu', dtype=torch.float32)
    buf360 = buf355; del buf355  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_53(c_void_p(buf360.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    del convolution_9
    del primals_19
    del relu_5
    del squeeze_28
    del unsqueeze_618
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf361 = aten.convolution_backward(buf360, add_47, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_47
    del buf360
    del primals_104
    buf362 = buf361[0]
    buf363 = buf361[1]
    del buf361
    buf364 = empty((24, ), device='cpu', dtype=torch.float32)
    buf365 = empty((24, ), device='cpu', dtype=torch.float32)
    buf366 = empty((24, ), device='cpu', dtype=torch.float32)
    buf367 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_54(c_void_p(buf362.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_630.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()))
    del convolution_8
    del primals_17
    del squeeze_25
    del unsqueeze_630
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf368 = aten.convolution_backward(buf367, relu_4, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf367
    del primals_103
    buf369 = buf368[0]
    buf370 = buf368[1]
    del buf368
    buf371 = buf358; del buf358  # reuse
    buf372 = empty((72, ), device='cpu', dtype=torch.float32)
    buf373 = empty((72, ), device='cpu', dtype=torch.float32)
    buf374 = buf369; del buf369  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_55(c_void_p(buf374.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_642.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()))
    del convolution_7
    del primals_15
    del relu_4
    del squeeze_22
    del unsqueeze_642
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf375 = aten.convolution_backward(buf374, relu_3, primals_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False])
    del buf374
    del primals_102
    buf376 = buf375[0]
    buf377 = buf375[1]
    del buf375
    buf378 = buf372; del buf372  # reuse
    buf379 = empty((72, ), device='cpu', dtype=torch.float32)
    buf380 = empty((72, ), device='cpu', dtype=torch.float32)
    buf381 = buf376; del buf376  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_56(c_void_p(buf381.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    del buf379
    del convolution_6
    del primals_13
    del relu_3
    del squeeze_19
    del unsqueeze_654
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf382 = aten.convolution_backward(buf381, add_31, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_31
    del buf381
    del primals_101
    buf383 = buf382[0]
    buf384 = buf382[1]
    del buf382
    buf385 = buf365; del buf365  # reuse
    buf386 = empty((24, ), device='cpu', dtype=torch.float32)
    buf387 = empty((24, ), device='cpu', dtype=torch.float32)
    buf388 = buf362; del buf362  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_57(c_void_p(buf388.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_666.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()))
    del buf383
    del buf386
    del convolution_5
    del primals_11
    del squeeze_16
    del unsqueeze_666
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf389 = aten.convolution_backward(buf388, relu_2, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf388
    del primals_100
    buf390 = buf389[0]
    buf391 = buf389[1]
    del buf389
    buf392 = empty((64, ), device='cpu', dtype=torch.float32)
    buf393 = empty((64, ), device='cpu', dtype=torch.float32)
    buf394 = empty((64, ), device='cpu', dtype=torch.float32)
    buf395 = buf390; del buf390  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_58(c_void_p(buf395.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_678.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    del convolution_4
    del primals_9
    del relu_2
    del squeeze_13
    del unsqueeze_678
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf396 = aten.convolution_backward(buf395, relu_1, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
    del primals_99
    buf397 = buf396[0]
    buf398 = buf396[1]
    del buf396
    buf399 = buf393; del buf393  # reuse
    buf400 = empty((64, ), device='cpu', dtype=torch.float32)
    buf401 = empty((64, ), device='cpu', dtype=torch.float32)
    buf402 = buf397; del buf397  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_59(c_void_p(buf402.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()))
    del buf400
    del convolution_3
    del primals_7
    del relu_1
    del squeeze_10
    del unsqueeze_690
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf403 = aten.convolution_backward(buf402, add_16, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_16
    del buf402
    del primals_98
    buf404 = buf403[0]
    buf405 = buf403[1]
    del buf403
    buf406 = empty((16, ), device='cpu', dtype=torch.float32)
    buf407 = empty((16, ), device='cpu', dtype=torch.float32)
    buf408 = empty((16, ), device='cpu', dtype=torch.float32)
    buf409 = reinterpret_tensor(buf395, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf395  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_60(c_void_p(buf404.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_702.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    del convolution_2
    del primals_5
    del squeeze_7
    del unsqueeze_702
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf410 = aten.convolution_backward(buf409, relu, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf409
    del primals_97
    buf411 = buf410[0]
    buf412 = buf410[1]
    del buf410
    buf413 = buf407; del buf407  # reuse
    buf414 = empty((16, ), device='cpu', dtype=torch.float32)
    buf415 = empty((16, ), device='cpu', dtype=torch.float32)
    buf416 = buf411; del buf411  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_61(c_void_p(buf416.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_714.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()))
    del convolution_1
    del primals_3
    del relu
    del squeeze_4
    del unsqueeze_714
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf417 = aten.convolution_backward(buf416, div, primals_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
    del buf416
    del div
    del primals_96
    buf418 = buf417[0]
    buf419 = buf417[1]
    del buf417
    buf420 = buf414; del buf414  # reuse
    buf421 = empty((16, ), device='cpu', dtype=torch.float32)
    buf422 = buf404; del buf404  # reuse
    buf423 = buf421; del buf421  # reuse
    cpp_fused_add_hardswish_backward_native_batch_norm_backward_62(c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(clone.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf420.data_ptr()))
    del buf418
    del clone
    del convolution
    del primals_1
    del squeeze_1
    del unsqueeze_726
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf424 = aten.convolution_backward(buf422, primals_313, primals_95, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf422
    del primals_313
    del primals_95
    buf425 = buf424[1]
    return (buf423, buf420, buf415, buf413, buf408, buf406, buf401, buf399, buf394, buf392, buf387, buf385, buf380, buf378, buf373, buf371, buf366, buf364, buf359, buf357, buf352, buf349, buf334, buf331, buf326, buf324, buf319, buf316, buf300, buf298, buf293, buf291, buf286, buf283, buf267, buf265, buf260, buf258, buf253, buf251, buf246, buf243, buf238, buf236, buf231, buf229, buf225, buf222, buf217, buf215, buf210, buf208, buf203, buf201, buf196, buf194, buf189, buf187, buf182, buf180, buf175, buf173, buf168, buf165, buf149, buf147, buf142, buf140, buf135, buf132, buf116, buf114, buf109, buf107, buf102, buf99, buf84, buf81, buf76, buf74, buf69, buf66, buf50, buf48, buf43, buf41, buf36, buf33, buf17, buf15, buf10, buf8, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), buf425, buf419, buf412, buf405, buf398, buf391, buf384, buf377, buf370, buf363, buf356, buf347, buf348, buf342, buf343, buf337, buf330, buf323, buf314, buf315, buf309, buf310, buf304, buf297, buf290, buf281, buf282, buf276, buf277, buf271, buf264, buf257, buf250, buf242, buf235, buf228, buf221, buf214, buf207, buf200, buf193, buf186, buf179, buf172, buf163, buf164, buf158, buf159, buf153, buf146, buf139, buf130, buf131, buf125, buf126, buf120, buf113, buf106, buf97, buf98, buf92, buf93, buf87, buf80, buf73, buf64, buf65, buf59, buf60, buf54, buf47, buf40, buf31, buf32, buf26, buf27, buf21, buf14, buf6, buf7, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    add_16 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_31 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_47 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    mul_78 = rand_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_63 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_100 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_80 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_122 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_97 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    clone_1 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    clone_2 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_114 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    clone_3 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    clone_4 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_132 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    clone_5 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    clone_6 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_150 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    clone_7 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    clone_8 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_168 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    clone_9 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    clone_10 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_238 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_186 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    clone_11 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    clone_12 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 168, 1, 1), (168, 1, 168, 168), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_262 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_205 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    clone_13 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    clone_14 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((8, 168, 1, 1), (168, 1, 168, 168), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_286 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    add_223 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clone_15 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clone_16 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    mul_310 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    add_242 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clone_17 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clone_18 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    mul_334 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    add_261 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clone_19 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 1280, 1, 1), (1280, 1, 1280, 1280), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((8, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    unsqueeze_210 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    unsqueeze_246 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_2 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    unsqueeze_282 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_3 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    unsqueeze_318 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_4 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.bool)
    unsqueeze_354 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_5 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    unsqueeze_534 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_6 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    unsqueeze_570 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_7 = rand_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.bool)
    unsqueeze_606 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_164, primals_165, primals_166, primals_167, primals_169, primals_171, primals_172, primals_173, primals_313, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, relu, convolution_2, squeeze_7, add_16, convolution_3, squeeze_10, relu_1, convolution_4, squeeze_13, relu_2, convolution_5, squeeze_16, add_31, convolution_6, squeeze_19, relu_3, convolution_7, squeeze_22, relu_4, convolution_8, squeeze_25, add_47, convolution_9, squeeze_28, relu_5, convolution_10, squeeze_31, relu_6, mean, relu_7, div_1, mul_78, convolution_13, squeeze_34, add_63, convolution_14, squeeze_37, relu_8, convolution_15, squeeze_40, relu_9, mean_1, relu_10, div_2, mul_100, convolution_18, squeeze_43, add_80, convolution_19, squeeze_46, relu_11, convolution_20, squeeze_49, relu_12, mean_2, relu_13, div_3, mul_122, convolution_23, squeeze_52, add_97, convolution_24, squeeze_55, clone_1, div_4, convolution_25, squeeze_58, clone_2, div_5, convolution_26, squeeze_61, add_114, convolution_27, squeeze_64, clone_3, div_6, convolution_28, squeeze_67, clone_4, div_7, convolution_29, squeeze_70, add_132, convolution_30, squeeze_73, clone_5, div_8, convolution_31, squeeze_76, clone_6, div_9, convolution_32, squeeze_79, add_150, convolution_33, squeeze_82, clone_7, div_10, convolution_34, squeeze_85, clone_8, div_11, convolution_35, squeeze_88, add_168, convolution_36, squeeze_91, clone_9, div_12, convolution_37, squeeze_94, clone_10, div_13, mean_3, relu_14, div_14, mul_238, convolution_40, squeeze_97, add_186, convolution_41, squeeze_100, clone_11, div_15, convolution_42, squeeze_103, clone_12, div_16, mean_4, relu_15, div_17, mul_262, convolution_45, squeeze_106, add_205, convolution_46, squeeze_109, clone_13, div_18, convolution_47, squeeze_112, clone_14, div_19, mean_5, relu_16, div_20, mul_286, convolution_50, squeeze_115, add_223, convolution_51, squeeze_118, clone_15, div_21, convolution_52, squeeze_121, clone_16, div_22, mean_6, relu_17, div_23, mul_310, convolution_55, squeeze_124, add_242, convolution_56, squeeze_127, clone_17, div_24, convolution_57, squeeze_130, clone_18, div_25, mean_7, relu_18, div_26, mul_334, convolution_60, squeeze_133, add_261, convolution_61, squeeze_136, clone_19, mean_8, convolution_62, view_1, permute_1, unsqueeze_186, unsqueeze_198, bitwise_and, unsqueeze_210, unsqueeze_222, unsqueeze_234, bitwise_and_1, unsqueeze_246, unsqueeze_258, unsqueeze_270, bitwise_and_2, unsqueeze_282, unsqueeze_294, unsqueeze_306, bitwise_and_3, unsqueeze_318, unsqueeze_330, unsqueeze_342, bitwise_and_4, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, unsqueeze_498, unsqueeze_510, unsqueeze_522, bitwise_and_5, unsqueeze_534, unsqueeze_546, unsqueeze_558, bitwise_and_6, unsqueeze_570, unsqueeze_582, unsqueeze_594, bitwise_and_7, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv3_large_100', benchmark_compiled_module)
