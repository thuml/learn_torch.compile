
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
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1408L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (1408L*x2) + (68992L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1408L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1408L*x2) + (68992L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1408L*x2) + (68992L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1408L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1408L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (1408L*x1) + (68992L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1408L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1408L*x1) + (68992L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1408L*x1) + (68992L*x0)));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(49.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.002551020408163265);
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
                        tmp25.store(out_ptr5 + static_cast<long>(x2 + (1408L*x1) + (68992L*x0)));
                        tmp39.store(out_ptr6 + static_cast<long>(x2 + (1408L*x1) + (68992L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1408L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_2 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp18 = tmp16 - tmp17;
                        auto tmp19 = tmp7 * tmp18;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                        tmp_acc2_vec = tmp_acc2_vec + tmp15;
                        tmp_acc3_vec = tmp_acc3_vec + tmp19;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp43 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp45 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp51 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp42 = tmp40 - tmp41;
                    auto tmp44 = tmp43 * tmp13;
                    auto tmp46 = tmp45 * tmp45;
                    auto tmp47 = tmp44 * tmp46;
                    auto tmp48 = tmp42 * tmp47;
                    auto tmp49 = tmp7 - tmp48;
                    auto tmp50 = tmp49 - tmp21;
                    auto tmp52 = tmp45 * tmp51;
                    auto tmp53 = tmp50 * tmp52;
                    tmp25.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    tmp39.store(out_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    tmp53.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr3;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp21 = tmp9 * tmp20;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                        tmp_acc2_vec = tmp_acc2_vec + tmp17;
                        tmp_acc3_vec = tmp_acc3_vec + tmp21;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp39 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp47 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp50 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                    auto tmp12 = tmp10 - tmp11;
                    auto tmp14 = static_cast<float>(0.0006377551020408163);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp18 = tmp17 * tmp17;
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp12 * tmp19;
                    auto tmp21 = tmp9 - tmp20;
                    auto tmp23 = tmp22 * tmp15;
                    auto tmp24 = tmp21 - tmp23;
                    auto tmp27 = tmp25 - tmp26;
                    auto tmp29 = tmp28 * tmp15;
                    auto tmp31 = tmp30 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    auto tmp33 = tmp27 * tmp32;
                    auto tmp34 = tmp9 - tmp33;
                    auto tmp35 = tmp34 - tmp23;
                    auto tmp38 = tmp36 - tmp37;
                    auto tmp40 = tmp39 * tmp15;
                    auto tmp42 = tmp41 * tmp41;
                    auto tmp43 = tmp40 * tmp42;
                    auto tmp44 = tmp38 * tmp43;
                    auto tmp45 = tmp9 - tmp44;
                    auto tmp46 = tmp45 - tmp23;
                    auto tmp48 = tmp17 * tmp47;
                    auto tmp49 = tmp24 * tmp48;
                    auto tmp51 = tmp30 * tmp50;
                    auto tmp52 = tmp35 * tmp51;
                    tmp46.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp49.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    tmp52.store(in_out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp10 = tmp8 * tmp9;
                    auto tmp11 = tmp7 * tmp10;
                    auto tmp12 = tmp6 + tmp11;
                    auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp0 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
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
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    tmp32.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(0.0006377551020408163);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp12 = tmp11 * tmp11;
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp15 = tmp3 - tmp14;
                    auto tmp17 = tmp16 * tmp9;
                    auto tmp18 = tmp15 - tmp17;
                    auto tmp20 = tmp11 * tmp19;
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp2 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp5 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp3 <= tmp5);
                    auto tmp8 = decltype(tmp5)::blendv(tmp7, tmp5, tmp6);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0006377551020408163);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp17 = tmp16 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp20 = tmp8 - tmp19;
                    auto tmp22 = tmp21 * tmp14;
                    auto tmp23 = tmp20 - tmp22;
                    auto tmp25 = tmp16 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp2 + tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0006377551020408163);
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    tmp37.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_30 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp18 = tmp16 - tmp17;
                        auto tmp19 = tmp7 * tmp18;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                        tmp_acc2_vec = tmp_acc2_vec + tmp15;
                        tmp_acc3_vec = tmp_acc3_vec + tmp19;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp43 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp45 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp51 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.00015943877551020407);
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
                    auto tmp42 = tmp40 - tmp41;
                    auto tmp44 = tmp43 * tmp13;
                    auto tmp46 = tmp45 * tmp45;
                    auto tmp47 = tmp44 * tmp46;
                    auto tmp48 = tmp42 * tmp47;
                    auto tmp49 = tmp7 - tmp48;
                    auto tmp50 = tmp49 - tmp21;
                    auto tmp52 = tmp45 * tmp51;
                    auto tmp53 = tmp50 * tmp52;
                    tmp25.store(out_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    tmp39.store(out_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    tmp53.store(out_ptr6 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_31 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr3;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp21 = tmp9 * tmp20;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                        tmp_acc2_vec = tmp_acc2_vec + tmp17;
                        tmp_acc3_vec = tmp_acc3_vec + tmp21;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp39 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp47 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp50 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                    auto tmp12 = tmp10 - tmp11;
                    auto tmp14 = static_cast<float>(0.00015943877551020407);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp18 = tmp17 * tmp17;
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp12 * tmp19;
                    auto tmp21 = tmp9 - tmp20;
                    auto tmp23 = tmp22 * tmp15;
                    auto tmp24 = tmp21 - tmp23;
                    auto tmp27 = tmp25 - tmp26;
                    auto tmp29 = tmp28 * tmp15;
                    auto tmp31 = tmp30 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    auto tmp33 = tmp27 * tmp32;
                    auto tmp34 = tmp9 - tmp33;
                    auto tmp35 = tmp34 - tmp23;
                    auto tmp38 = tmp36 - tmp37;
                    auto tmp40 = tmp39 * tmp15;
                    auto tmp42 = tmp41 * tmp41;
                    auto tmp43 = tmp40 * tmp42;
                    auto tmp44 = tmp38 * tmp43;
                    auto tmp45 = tmp9 - tmp44;
                    auto tmp46 = tmp45 - tmp23;
                    auto tmp48 = tmp17 * tmp47;
                    auto tmp49 = tmp24 * tmp48;
                    auto tmp51 = tmp30 * tmp50;
                    auto tmp52 = tmp35 * tmp51;
                    tmp46.store(out_ptr6 + static_cast<long>(x1 + (192L*x0)));
                    tmp49.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    tmp52.store(in_out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_33 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.cpp('''
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp10 = tmp8 * tmp9;
                    auto tmp11 = tmp7 * tmp10;
                    auto tmp12 = tmp6 + tmp11;
                    auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp0 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
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
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    tmp32.store(out_ptr6 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_35 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(0.00015943877551020407);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp12 = tmp11 * tmp11;
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp15 = tmp3 - tmp14;
                    auto tmp17 = tmp16 * tmp9;
                    auto tmp18 = tmp15 - tmp17;
                    auto tmp20 = tmp11 * tmp19;
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp2 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1));
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
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = tmp27 * tmp11;
                    auto tmp30 = tmp29 * tmp29;
                    auto tmp31 = tmp28 * tmp30;
                    auto tmp32 = tmp26 * tmp31;
                    auto tmp33 = tmp5 - tmp32;
                    auto tmp34 = tmp33 - tmp19;
                    auto tmp36 = tmp29 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp23.store(out_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    tmp37.store(out_ptr6 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_37 = async_compile.cpp('''
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
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
                       const float* in_ptr13,
                       const float* in_ptr14,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp18 = tmp16 - tmp17;
                        auto tmp19 = tmp7 * tmp18;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                        tmp_acc2_vec = tmp_acc2_vec + tmp15;
                        tmp_acc3_vec = tmp_acc3_vec + tmp19;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp43 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp45 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp51 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(3.985969387755102e-05);
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
                    auto tmp42 = tmp40 - tmp41;
                    auto tmp44 = tmp43 * tmp13;
                    auto tmp46 = tmp45 * tmp45;
                    auto tmp47 = tmp44 * tmp46;
                    auto tmp48 = tmp42 * tmp47;
                    auto tmp49 = tmp7 - tmp48;
                    auto tmp50 = tmp49 - tmp21;
                    auto tmp52 = tmp45 * tmp51;
                    auto tmp53 = tmp50 * tmp52;
                    tmp25.store(out_ptr4 + static_cast<long>(x1 + (96L*x0)));
                    tmp39.store(out_ptr5 + static_cast<long>(x1 + (96L*x0)));
                    tmp53.store(out_ptr6 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr3;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp17 = tmp9 * tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
                        tmp_acc2_vec = tmp_acc2_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp39 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                    auto tmp12 = tmp10 - tmp11;
                    auto tmp14 = static_cast<float>(3.985969387755102e-05);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp18 = tmp17 * tmp17;
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp12 * tmp19;
                    auto tmp21 = tmp9 - tmp20;
                    auto tmp23 = tmp22 * tmp15;
                    auto tmp24 = tmp21 - tmp23;
                    auto tmp27 = tmp25 - tmp26;
                    auto tmp29 = tmp28 * tmp15;
                    auto tmp31 = tmp30 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    auto tmp33 = tmp27 * tmp32;
                    auto tmp34 = tmp9 - tmp33;
                    auto tmp35 = tmp34 - tmp23;
                    auto tmp37 = tmp17 * tmp36;
                    auto tmp38 = tmp24 * tmp37;
                    auto tmp40 = tmp30 * tmp39;
                    auto tmp41 = tmp35 * tmp40;
                    tmp38.store(in_out_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    tmp41.store(in_out_ptr2 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (64L*x0)));
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
                    auto tmp12 = static_cast<float>(9.964923469387754e-06);
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
                    tmp25.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    tmp39.store(out_ptr4 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_352, convolution, squeeze_1, convolution_1, squeeze_4, relu, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_1, squeeze_13, convolution_4, squeeze_16, convolution_5, squeeze_19, relu_2, convolution_6, squeeze_22, convolution_7, squeeze_25, relu_3, squeeze_28, convolution_8, squeeze_31, convolution_9, squeeze_34, relu_4, squeeze_37, convolution_10, squeeze_40, convolution_11, squeeze_43, relu_5, squeeze_46, convolution_12, squeeze_49, convolution_13, squeeze_52, relu_6, convolution_14, squeeze_55, convolution_15, squeeze_58, relu_7, squeeze_61, convolution_16, squeeze_64, convolution_17, squeeze_67, relu_8, squeeze_70, convolution_18, squeeze_73, convolution_19, squeeze_76, relu_9, squeeze_79, convolution_20, squeeze_82, convolution_21, squeeze_85, relu_10, squeeze_88, convolution_22, squeeze_91, convolution_23, squeeze_94, relu_11, squeeze_97, convolution_24, squeeze_100, convolution_25, squeeze_103, relu_12, squeeze_106, convolution_26, squeeze_109, convolution_27, squeeze_112, relu_13, squeeze_115, convolution_28, squeeze_118, convolution_29, squeeze_121, relu_14, squeeze_124, convolution_30, squeeze_127, convolution_31, squeeze_130, relu_15, squeeze_133, convolution_32, squeeze_136, convolution_33, squeeze_139, relu_16, squeeze_142, convolution_34, squeeze_145, convolution_35, squeeze_148, relu_17, squeeze_151, convolution_36, squeeze_154, convolution_37, squeeze_157, relu_18, squeeze_160, convolution_38, squeeze_163, convolution_39, squeeze_166, relu_19, squeeze_169, convolution_40, squeeze_172, convolution_41, squeeze_175, relu_20, convolution_42, squeeze_178, convolution_43, squeeze_181, clone, permute_1, le, unsqueeze_246, unsqueeze_258, unsqueeze_270, unsqueeze_282, unsqueeze_294, unsqueeze_306, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, unsqueeze_498, unsqueeze_510, unsqueeze_522, unsqueeze_534, unsqueeze_546, unsqueeze_558, unsqueeze_570, unsqueeze_582, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, unsqueeze_738, unsqueeze_750, unsqueeze_762, unsqueeze_774, unsqueeze_786, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, unsqueeze_858, unsqueeze_870, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (96, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_13, (96, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_19, (192, ), (1, ))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_23, (192, ), (1, ))
    assert_size_stride(primals_25, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_37, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_47, (384, ), (1, ))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_53, (384, ), (1, ))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_59, (384, ), (1, ))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_71, (384, ), (1, ))
    assert_size_stride(primals_73, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_77, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_97, (384, ), (1, ))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_101, (384, ), (1, ))
    assert_size_stride(primals_103, (384, ), (1, ))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_109, (384, ), (1, ))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_113, (384, ), (1, ))
    assert_size_stride(primals_115, (384, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_119, (1408, ), (1, ))
    assert_size_stride(primals_121, (1408, ), (1, ))
    assert_size_stride(primals_123, (64, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_124, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_125, (96, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_126, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_127, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_128, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_129, (192, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_130, (192, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_131, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_132, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_133, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_134, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_135, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_136, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_137, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_138, (384, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_139, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_140, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_141, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_142, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_143, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_144, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_145, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_146, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_147, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_148, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_149, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_150, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_151, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_152, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_153, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_154, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_155, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_156, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_157, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_158, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_159, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_160, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_161, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_162, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_163, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_164, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_165, (1408, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_166, (1408, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_352, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(convolution_1, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_2, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_7, (96, ), (1, ))
    assert_size_stride(convolution_3, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(relu_1, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(convolution_4, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_16, (96, ), (1, ))
    assert_size_stride(convolution_5, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_19, (96, ), (1, ))
    assert_size_stride(relu_2, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_6, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_22, (192, ), (1, ))
    assert_size_stride(convolution_7, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_25, (192, ), (1, ))
    assert_size_stride(relu_3, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_28, (192, ), (1, ))
    assert_size_stride(convolution_8, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_31, (192, ), (1, ))
    assert_size_stride(convolution_9, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_34, (192, ), (1, ))
    assert_size_stride(relu_4, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_37, (192, ), (1, ))
    assert_size_stride(convolution_10, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_40, (192, ), (1, ))
    assert_size_stride(convolution_11, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_43, (192, ), (1, ))
    assert_size_stride(relu_5, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_46, (192, ), (1, ))
    assert_size_stride(convolution_12, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_49, (192, ), (1, ))
    assert_size_stride(convolution_13, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_52, (192, ), (1, ))
    assert_size_stride(relu_6, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_14, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_55, (384, ), (1, ))
    assert_size_stride(convolution_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_58, (384, ), (1, ))
    assert_size_stride(relu_7, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_61, (384, ), (1, ))
    assert_size_stride(convolution_16, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_64, (384, ), (1, ))
    assert_size_stride(convolution_17, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_67, (384, ), (1, ))
    assert_size_stride(relu_8, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_70, (384, ), (1, ))
    assert_size_stride(convolution_18, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_73, (384, ), (1, ))
    assert_size_stride(convolution_19, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_76, (384, ), (1, ))
    assert_size_stride(relu_9, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_79, (384, ), (1, ))
    assert_size_stride(convolution_20, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_82, (384, ), (1, ))
    assert_size_stride(convolution_21, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_85, (384, ), (1, ))
    assert_size_stride(relu_10, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_88, (384, ), (1, ))
    assert_size_stride(convolution_22, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_91, (384, ), (1, ))
    assert_size_stride(convolution_23, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_94, (384, ), (1, ))
    assert_size_stride(relu_11, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_97, (384, ), (1, ))
    assert_size_stride(convolution_24, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_100, (384, ), (1, ))
    assert_size_stride(convolution_25, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_103, (384, ), (1, ))
    assert_size_stride(relu_12, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_106, (384, ), (1, ))
    assert_size_stride(convolution_26, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_109, (384, ), (1, ))
    assert_size_stride(convolution_27, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_112, (384, ), (1, ))
    assert_size_stride(relu_13, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_115, (384, ), (1, ))
    assert_size_stride(convolution_28, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_118, (384, ), (1, ))
    assert_size_stride(convolution_29, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_121, (384, ), (1, ))
    assert_size_stride(relu_14, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_124, (384, ), (1, ))
    assert_size_stride(convolution_30, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_127, (384, ), (1, ))
    assert_size_stride(convolution_31, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_130, (384, ), (1, ))
    assert_size_stride(relu_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_133, (384, ), (1, ))
    assert_size_stride(convolution_32, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_136, (384, ), (1, ))
    assert_size_stride(convolution_33, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_139, (384, ), (1, ))
    assert_size_stride(relu_16, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_142, (384, ), (1, ))
    assert_size_stride(convolution_34, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_145, (384, ), (1, ))
    assert_size_stride(convolution_35, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_148, (384, ), (1, ))
    assert_size_stride(relu_17, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_151, (384, ), (1, ))
    assert_size_stride(convolution_36, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_154, (384, ), (1, ))
    assert_size_stride(convolution_37, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_157, (384, ), (1, ))
    assert_size_stride(relu_18, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_160, (384, ), (1, ))
    assert_size_stride(convolution_38, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_163, (384, ), (1, ))
    assert_size_stride(convolution_39, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_166, (384, ), (1, ))
    assert_size_stride(relu_19, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_169, (384, ), (1, ))
    assert_size_stride(convolution_40, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_172, (384, ), (1, ))
    assert_size_stride(convolution_41, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_175, (384, ), (1, ))
    assert_size_stride(relu_20, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_42, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
    assert_size_stride(squeeze_178, (1408, ), (1, ))
    assert_size_stride(convolution_43, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
    assert_size_stride(squeeze_181, (1408, ), (1, ))
    assert_size_stride(clone, (8, 1408), (1408, 1))
    assert_size_stride(permute_1, (1000, 1408), (1408, 1))
    assert_size_stride(le, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
    assert_size_stride(unsqueeze_246, (1, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(unsqueeze_270, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_318, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_342, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_486, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_522, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_558, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_594, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_630, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_666, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_702, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_738, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_774, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_786, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_798, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_810, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_822, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_846, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_858, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_870, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_882, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_894, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_906, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_918, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_930, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_942, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_954, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_966, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1408), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1408), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
    del clone
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((1408, ), device='cpu', dtype=torch.float32)
    buf4 = empty((1408, ), device='cpu', dtype=torch.float32)
    buf10 = empty((1408, ), device='cpu', dtype=torch.float32)
    buf5 = empty((1408, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf12.data_ptr()))
    del buf0
    del buf4
    del convolution_42
    del convolution_43
    del le
    del primals_119
    del primals_121
    del squeeze_181
    del tangents_1
    del unsqueeze_246
    del unsqueeze_258
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, relu_20, primals_166, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf6
    del primals_166
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf11 = buf10; del buf10  # reuse
    cpp_fused_native_batch_norm_backward_1(c_void_p(buf11.data_ptr()), c_void_p(squeeze_178.data_ptr()))
    del squeeze_178
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf13 = aten.convolution_backward(buf12, relu_20, primals_165, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf12
    del primals_165
    buf14 = buf13[0]
    buf15 = buf13[1]
    del buf13
    buf16 = empty((384, ), device='cpu', dtype=torch.float32)
    buf17 = empty((384, ), device='cpu', dtype=torch.float32)
    buf23 = empty((384, ), device='cpu', dtype=torch.float32)
    buf29 = empty((384, ), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf19 = buf17; del buf17  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_2(c_void_p(buf19.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_270.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(unsqueeze_294.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf30.data_ptr()))
    del buf14
    del convolution_40
    del convolution_41
    del primals_113
    del primals_115
    del primals_117
    del relu_20
    del squeeze_175
    del unsqueeze_270
    del unsqueeze_282
    del unsqueeze_294
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf20 = aten.convolution_backward(buf18, relu_19, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_164
    buf21 = buf20[0]
    buf22 = buf20[1]
    del buf20
    buf25 = buf23; del buf23  # reuse
    cpp_fused_native_batch_norm_backward_3(c_void_p(buf25.data_ptr()), c_void_p(squeeze_172.data_ptr()))
    del squeeze_172
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf26 = aten.convolution_backward(buf24, relu_19, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_163
    buf27 = buf26[0]
    buf28 = buf26[1]
    del buf26
    buf31 = buf29; del buf29  # reuse
    buf32 = empty((384, ), device='cpu', dtype=torch.float32)
    buf33 = empty((384, ), device='cpu', dtype=torch.float32)
    buf40 = empty((384, ), device='cpu', dtype=torch.float32)
    buf47 = empty((384, ), device='cpu', dtype=torch.float32)
    buf34 = buf24; del buf24  # reuse
    buf41 = buf18; del buf18  # reuse
    buf48 = buf8; del buf8  # reuse
    buf36 = buf34; del buf34  # reuse
    buf43 = buf41; del buf41  # reuse
    buf35 = buf33; del buf33  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf31.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_318.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del buf21
    del buf27
    del buf30
    del convolution_38
    del convolution_39
    del primals_109
    del primals_111
    del relu_19
    del squeeze_166
    del squeeze_169
    del unsqueeze_306
    del unsqueeze_318
    del unsqueeze_330
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf37 = aten.convolution_backward(buf36, relu_18, primals_162, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_162
    buf38 = buf37[0]
    buf39 = buf37[1]
    del buf37
    buf42 = buf40; del buf40  # reuse
    cpp_fused_native_batch_norm_backward_5(c_void_p(buf42.data_ptr()), c_void_p(squeeze_163.data_ptr()))
    del squeeze_163
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf44 = aten.convolution_backward(buf43, relu_18, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_161
    buf45 = buf44[0]
    buf46 = buf44[1]
    del buf44
    buf49 = buf47; del buf47  # reuse
    buf50 = buf38; del buf38  # reuse
    buf51 = empty((384, ), device='cpu', dtype=torch.float32)
    buf52 = empty((384, ), device='cpu', dtype=torch.float32)
    buf58 = empty((384, ), device='cpu', dtype=torch.float32)
    buf64 = empty((384, ), device='cpu', dtype=torch.float32)
    buf53 = empty((384, ), device='cpu', dtype=torch.float32)
    buf54 = buf43; del buf43  # reuse
    buf60 = buf36; del buf36  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf60.data_ptr()))
    del buf45
    del buf48
    del convolution_36
    del convolution_37
    del primals_103
    del primals_105
    del primals_107
    del relu_18
    del squeeze_157
    del squeeze_160
    del unsqueeze_342
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf55 = aten.convolution_backward(buf54, relu_17, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_160
    buf56 = buf55[0]
    buf57 = buf55[1]
    del buf55
    buf59 = buf58; del buf58  # reuse
    cpp_fused_native_batch_norm_backward_7(c_void_p(buf59.data_ptr()), c_void_p(squeeze_154.data_ptr()))
    del squeeze_154
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf61 = aten.convolution_backward(buf60, relu_17, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_159
    buf62 = buf61[0]
    buf63 = buf61[1]
    del buf61
    buf65 = buf52; del buf52  # reuse
    buf66 = buf50; del buf50  # reuse
    buf67 = empty((384, ), device='cpu', dtype=torch.float32)
    buf68 = empty((384, ), device='cpu', dtype=torch.float32)
    buf74 = empty((384, ), device='cpu', dtype=torch.float32)
    buf80 = empty((384, ), device='cpu', dtype=torch.float32)
    buf69 = empty((384, ), device='cpu', dtype=torch.float32)
    buf70 = buf60; del buf60  # reuse
    buf76 = buf54; del buf54  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8(c_void_p(buf66.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf76.data_ptr()))
    del buf56
    del buf62
    del convolution_34
    del convolution_35
    del primals_101
    del primals_97
    del primals_99
    del squeeze_148
    del squeeze_151
    del unsqueeze_366
    del unsqueeze_378
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf71 = aten.convolution_backward(buf70, relu_16, primals_158, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_158
    buf72 = buf71[0]
    buf73 = buf71[1]
    del buf71
    buf75 = buf74; del buf74  # reuse
    cpp_fused_native_batch_norm_backward_9(c_void_p(buf75.data_ptr()), c_void_p(squeeze_145.data_ptr()))
    del squeeze_145
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf77 = aten.convolution_backward(buf76, relu_16, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_157
    buf78 = buf77[0]
    buf79 = buf77[1]
    del buf77
    buf81 = buf68; del buf68  # reuse
    buf82 = buf66; del buf66  # reuse
    buf83 = buf64; del buf64  # reuse
    buf84 = empty((384, ), device='cpu', dtype=torch.float32)
    buf90 = empty((384, ), device='cpu', dtype=torch.float32)
    buf96 = empty((384, ), device='cpu', dtype=torch.float32)
    buf85 = empty((384, ), device='cpu', dtype=torch.float32)
    buf86 = buf76; del buf76  # reuse
    buf92 = buf70; del buf70  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf82.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf92.data_ptr()))
    del buf72
    del buf78
    del convolution_32
    del convolution_33
    del primals_91
    del primals_93
    del primals_95
    del relu_17
    del squeeze_139
    del squeeze_142
    del unsqueeze_402
    del unsqueeze_414
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf87 = aten.convolution_backward(buf86, relu_15, primals_156, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_156
    buf88 = buf87[0]
    buf89 = buf87[1]
    del buf87
    buf91 = buf90; del buf90  # reuse
    cpp_fused_native_batch_norm_backward_11(c_void_p(buf91.data_ptr()), c_void_p(squeeze_136.data_ptr()))
    del squeeze_136
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf93 = aten.convolution_backward(buf92, relu_15, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_155
    buf94 = buf93[0]
    buf95 = buf93[1]
    del buf93
    buf97 = buf84; del buf84  # reuse
    buf98 = buf82; del buf82  # reuse
    buf99 = buf80; del buf80  # reuse
    buf100 = empty((384, ), device='cpu', dtype=torch.float32)
    buf106 = empty((384, ), device='cpu', dtype=torch.float32)
    buf112 = empty((384, ), device='cpu', dtype=torch.float32)
    buf101 = empty((384, ), device='cpu', dtype=torch.float32)
    buf102 = buf92; del buf92  # reuse
    buf108 = buf86; del buf86  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf98.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf108.data_ptr()))
    del buf88
    del buf94
    del convolution_30
    del convolution_31
    del primals_85
    del primals_87
    del primals_89
    del relu_16
    del squeeze_130
    del squeeze_133
    del unsqueeze_438
    del unsqueeze_450
    del unsqueeze_462
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf103 = aten.convolution_backward(buf102, relu_14, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_154
    buf104 = buf103[0]
    buf105 = buf103[1]
    del buf103
    buf107 = buf106; del buf106  # reuse
    cpp_fused_native_batch_norm_backward_13(c_void_p(buf107.data_ptr()), c_void_p(squeeze_127.data_ptr()))
    del squeeze_127
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf109 = aten.convolution_backward(buf108, relu_14, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_153
    buf110 = buf109[0]
    buf111 = buf109[1]
    del buf109
    buf113 = buf96; del buf96  # reuse
    buf114 = buf104; del buf104  # reuse
    buf115 = buf100; del buf100  # reuse
    buf116 = empty((384, ), device='cpu', dtype=torch.float32)
    buf122 = empty((384, ), device='cpu', dtype=torch.float32)
    buf128 = empty((384, ), device='cpu', dtype=torch.float32)
    buf117 = empty((384, ), device='cpu', dtype=torch.float32)
    buf118 = buf108; del buf108  # reuse
    buf124 = buf102; del buf102  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf114.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf124.data_ptr()))
    del buf110
    del buf98
    del convolution_28
    del convolution_29
    del primals_79
    del primals_81
    del primals_83
    del relu_15
    del squeeze_121
    del squeeze_124
    del unsqueeze_474
    del unsqueeze_486
    del unsqueeze_498
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf119 = aten.convolution_backward(buf118, relu_13, primals_152, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_152
    buf120 = buf119[0]
    buf121 = buf119[1]
    del buf119
    buf123 = buf122; del buf122  # reuse
    cpp_fused_native_batch_norm_backward_15(c_void_p(buf123.data_ptr()), c_void_p(squeeze_118.data_ptr()))
    del squeeze_118
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf125 = aten.convolution_backward(buf124, relu_13, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_151
    buf126 = buf125[0]
    buf127 = buf125[1]
    del buf125
    buf129 = buf116; del buf116  # reuse
    buf130 = buf114; del buf114  # reuse
    buf131 = buf112; del buf112  # reuse
    buf132 = empty((384, ), device='cpu', dtype=torch.float32)
    buf138 = empty((384, ), device='cpu', dtype=torch.float32)
    buf144 = empty((384, ), device='cpu', dtype=torch.float32)
    buf133 = empty((384, ), device='cpu', dtype=torch.float32)
    buf134 = buf124; del buf124  # reuse
    buf140 = buf118; del buf118  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_16(c_void_p(buf130.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_522.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_534.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf140.data_ptr()))
    del buf120
    del buf126
    del convolution_26
    del convolution_27
    del primals_73
    del primals_75
    del primals_77
    del relu_14
    del squeeze_112
    del squeeze_115
    del unsqueeze_510
    del unsqueeze_522
    del unsqueeze_534
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf135 = aten.convolution_backward(buf134, relu_12, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_150
    buf136 = buf135[0]
    buf137 = buf135[1]
    del buf135
    buf139 = buf138; del buf138  # reuse
    cpp_fused_native_batch_norm_backward_17(c_void_p(buf139.data_ptr()), c_void_p(squeeze_109.data_ptr()))
    del squeeze_109
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf141 = aten.convolution_backward(buf140, relu_12, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_149
    buf142 = buf141[0]
    buf143 = buf141[1]
    del buf141
    buf145 = buf132; del buf132  # reuse
    buf146 = buf130; del buf130  # reuse
    buf147 = buf128; del buf128  # reuse
    buf148 = empty((384, ), device='cpu', dtype=torch.float32)
    buf154 = empty((384, ), device='cpu', dtype=torch.float32)
    buf160 = empty((384, ), device='cpu', dtype=torch.float32)
    buf149 = empty((384, ), device='cpu', dtype=torch.float32)
    buf150 = buf140; del buf140  # reuse
    buf156 = buf134; del buf134  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(buf146.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_558.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_570.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf156.data_ptr()))
    del buf136
    del buf142
    del convolution_24
    del convolution_25
    del primals_67
    del primals_69
    del primals_71
    del relu_13
    del squeeze_103
    del squeeze_106
    del unsqueeze_546
    del unsqueeze_558
    del unsqueeze_570
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf151 = aten.convolution_backward(buf150, relu_11, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_148
    buf152 = buf151[0]
    buf153 = buf151[1]
    del buf151
    buf155 = buf154; del buf154  # reuse
    cpp_fused_native_batch_norm_backward_19(c_void_p(buf155.data_ptr()), c_void_p(squeeze_100.data_ptr()))
    del squeeze_100
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf157 = aten.convolution_backward(buf156, relu_11, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_147
    buf158 = buf157[0]
    buf159 = buf157[1]
    del buf157
    buf161 = buf148; del buf148  # reuse
    buf162 = buf146; del buf146  # reuse
    buf163 = buf144; del buf144  # reuse
    buf164 = empty((384, ), device='cpu', dtype=torch.float32)
    buf170 = empty((384, ), device='cpu', dtype=torch.float32)
    buf176 = empty((384, ), device='cpu', dtype=torch.float32)
    buf165 = empty((384, ), device='cpu', dtype=torch.float32)
    buf166 = buf156; del buf156  # reuse
    buf172 = buf150; del buf150  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(buf162.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_594.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_606.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf172.data_ptr()))
    del buf152
    del buf158
    del convolution_22
    del convolution_23
    del primals_61
    del primals_63
    del primals_65
    del relu_12
    del squeeze_94
    del squeeze_97
    del unsqueeze_582
    del unsqueeze_594
    del unsqueeze_606
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf167 = aten.convolution_backward(buf166, relu_10, primals_146, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_146
    buf168 = buf167[0]
    buf169 = buf167[1]
    del buf167
    buf171 = buf170; del buf170  # reuse
    cpp_fused_native_batch_norm_backward_21(c_void_p(buf171.data_ptr()), c_void_p(squeeze_91.data_ptr()))
    del squeeze_91
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf173 = aten.convolution_backward(buf172, relu_10, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_145
    buf174 = buf173[0]
    buf175 = buf173[1]
    del buf173
    buf177 = buf164; del buf164  # reuse
    buf178 = buf162; del buf162  # reuse
    buf179 = buf160; del buf160  # reuse
    buf180 = empty((384, ), device='cpu', dtype=torch.float32)
    buf186 = empty((384, ), device='cpu', dtype=torch.float32)
    buf192 = empty((384, ), device='cpu', dtype=torch.float32)
    buf181 = empty((384, ), device='cpu', dtype=torch.float32)
    buf182 = buf172; del buf172  # reuse
    buf188 = buf166; del buf166  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(buf178.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_630.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_642.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf188.data_ptr()))
    del buf168
    del buf174
    del convolution_20
    del convolution_21
    del primals_55
    del primals_57
    del primals_59
    del relu_11
    del squeeze_85
    del squeeze_88
    del unsqueeze_618
    del unsqueeze_630
    del unsqueeze_642
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf183 = aten.convolution_backward(buf182, relu_9, primals_144, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_144
    buf184 = buf183[0]
    buf185 = buf183[1]
    del buf183
    buf187 = buf186; del buf186  # reuse
    cpp_fused_native_batch_norm_backward_23(c_void_p(buf187.data_ptr()), c_void_p(squeeze_82.data_ptr()))
    del squeeze_82
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf189 = aten.convolution_backward(buf188, relu_9, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_143
    buf190 = buf189[0]
    buf191 = buf189[1]
    del buf189
    buf193 = buf180; del buf180  # reuse
    buf194 = buf178; del buf178  # reuse
    buf195 = buf176; del buf176  # reuse
    buf196 = empty((384, ), device='cpu', dtype=torch.float32)
    buf202 = empty((384, ), device='cpu', dtype=torch.float32)
    buf208 = empty((384, ), device='cpu', dtype=torch.float32)
    buf197 = empty((384, ), device='cpu', dtype=torch.float32)
    buf198 = buf188; del buf188  # reuse
    buf204 = buf182; del buf182  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf194.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_666.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_678.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf204.data_ptr()))
    del buf184
    del buf190
    del convolution_18
    del convolution_19
    del primals_49
    del primals_51
    del primals_53
    del relu_10
    del squeeze_76
    del squeeze_79
    del unsqueeze_654
    del unsqueeze_666
    del unsqueeze_678
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf199 = aten.convolution_backward(buf198, relu_8, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_142
    buf200 = buf199[0]
    buf201 = buf199[1]
    del buf199
    buf203 = buf202; del buf202  # reuse
    cpp_fused_native_batch_norm_backward_25(c_void_p(buf203.data_ptr()), c_void_p(squeeze_73.data_ptr()))
    del squeeze_73
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf205 = aten.convolution_backward(buf204, relu_8, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_141
    buf206 = buf205[0]
    buf207 = buf205[1]
    del buf205
    buf209 = buf196; del buf196  # reuse
    buf210 = buf194; del buf194  # reuse
    buf211 = buf192; del buf192  # reuse
    buf212 = empty((384, ), device='cpu', dtype=torch.float32)
    buf218 = empty((384, ), device='cpu', dtype=torch.float32)
    buf224 = empty((384, ), device='cpu', dtype=torch.float32)
    buf213 = empty((384, ), device='cpu', dtype=torch.float32)
    buf214 = buf204; del buf204  # reuse
    buf220 = buf198; del buf198  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_26(c_void_p(buf210.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_702.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_714.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf220.data_ptr()))
    del buf200
    del buf206
    del convolution_16
    del convolution_17
    del primals_43
    del primals_45
    del primals_47
    del relu_9
    del squeeze_67
    del squeeze_70
    del unsqueeze_690
    del unsqueeze_702
    del unsqueeze_714
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf215 = aten.convolution_backward(buf214, relu_7, primals_140, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_140
    buf216 = buf215[0]
    buf217 = buf215[1]
    del buf215
    buf219 = buf218; del buf218  # reuse
    cpp_fused_native_batch_norm_backward_27(c_void_p(buf219.data_ptr()), c_void_p(squeeze_64.data_ptr()))
    del squeeze_64
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf221 = aten.convolution_backward(buf220, relu_7, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_139
    buf222 = buf221[0]
    buf223 = buf221[1]
    del buf221
    buf225 = buf212; del buf212  # reuse
    buf226 = buf210; del buf210  # reuse
    buf227 = buf208; del buf208  # reuse
    buf228 = empty((384, ), device='cpu', dtype=torch.float32)
    buf234 = empty((384, ), device='cpu', dtype=torch.float32)
    buf229 = empty((384, ), device='cpu', dtype=torch.float32)
    buf230 = buf220; del buf220  # reuse
    buf236 = buf214; del buf214  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_28(c_void_p(buf226.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_738.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_750.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf236.data_ptr()))
    del buf216
    del buf222
    del buf224
    del buf226
    del buf228
    del convolution_14
    del convolution_15
    del primals_37
    del primals_39
    del primals_41
    del relu_7
    del relu_8
    del squeeze_58
    del squeeze_61
    del unsqueeze_726
    del unsqueeze_738
    del unsqueeze_750
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf231 = aten.convolution_backward(buf230, relu_6, primals_138, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf230
    del primals_138
    buf232 = buf231[0]
    buf233 = buf231[1]
    del buf231
    buf235 = buf234; del buf234  # reuse
    cpp_fused_native_batch_norm_backward_29(c_void_p(buf235.data_ptr()), c_void_p(squeeze_55.data_ptr()))
    del squeeze_55
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf237 = aten.convolution_backward(buf236, relu_6, primals_137, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf236
    del primals_137
    buf238 = buf237[0]
    buf239 = buf237[1]
    del buf237
    buf240 = empty((192, ), device='cpu', dtype=torch.float32)
    buf241 = empty((192, ), device='cpu', dtype=torch.float32)
    buf247 = empty((192, ), device='cpu', dtype=torch.float32)
    buf253 = empty((192, ), device='cpu', dtype=torch.float32)
    buf242 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf248 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf254 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf243 = buf241; del buf241  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_30(c_void_p(buf243.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_762.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_774.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(unsqueeze_786.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf254.data_ptr()))
    del buf232
    del convolution_12
    del convolution_13
    del primals_31
    del primals_33
    del primals_35
    del relu_6
    del squeeze_52
    del unsqueeze_762
    del unsqueeze_774
    del unsqueeze_786
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf244 = aten.convolution_backward(buf242, relu_5, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_136
    buf245 = buf244[0]
    buf246 = buf244[1]
    del buf244
    buf249 = buf247; del buf247  # reuse
    cpp_fused_native_batch_norm_backward_31(c_void_p(buf249.data_ptr()), c_void_p(squeeze_49.data_ptr()))
    del squeeze_49
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf250 = aten.convolution_backward(buf248, relu_5, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_135
    buf251 = buf250[0]
    buf252 = buf250[1]
    del buf250
    buf255 = buf253; del buf253  # reuse
    buf256 = empty((192, ), device='cpu', dtype=torch.float32)
    buf257 = empty((192, ), device='cpu', dtype=torch.float32)
    buf264 = empty((192, ), device='cpu', dtype=torch.float32)
    buf271 = empty((192, ), device='cpu', dtype=torch.float32)
    buf258 = buf248; del buf248  # reuse
    buf265 = buf242; del buf242  # reuse
    buf272 = buf238; del buf238  # reuse
    buf260 = buf258; del buf258  # reuse
    buf267 = buf265; del buf265  # reuse
    buf259 = buf257; del buf257  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(buf255.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_798.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_810.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(unsqueeze_822.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    del buf245
    del buf251
    del buf254
    del convolution_10
    del convolution_11
    del primals_27
    del primals_29
    del relu_5
    del squeeze_43
    del squeeze_46
    del unsqueeze_798
    del unsqueeze_810
    del unsqueeze_822
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf261 = aten.convolution_backward(buf260, relu_4, primals_134, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_134
    buf262 = buf261[0]
    buf263 = buf261[1]
    del buf261
    buf266 = buf264; del buf264  # reuse
    cpp_fused_native_batch_norm_backward_33(c_void_p(buf266.data_ptr()), c_void_p(squeeze_40.data_ptr()))
    del squeeze_40
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf268 = aten.convolution_backward(buf267, relu_4, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_133
    buf269 = buf268[0]
    buf270 = buf268[1]
    del buf268
    buf273 = buf271; del buf271  # reuse
    buf274 = buf262; del buf262  # reuse
    buf275 = empty((192, ), device='cpu', dtype=torch.float32)
    buf276 = empty((192, ), device='cpu', dtype=torch.float32)
    buf282 = empty((192, ), device='cpu', dtype=torch.float32)
    buf288 = empty((192, ), device='cpu', dtype=torch.float32)
    buf277 = empty((192, ), device='cpu', dtype=torch.float32)
    buf278 = buf267; del buf267  # reuse
    buf284 = buf260; del buf260  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_34(c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_834.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_846.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(unsqueeze_858.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf284.data_ptr()))
    del buf269
    del buf272
    del convolution_8
    del convolution_9
    del primals_21
    del primals_23
    del primals_25
    del relu_4
    del squeeze_34
    del squeeze_37
    del unsqueeze_834
    del unsqueeze_846
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf279 = aten.convolution_backward(buf278, relu_3, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_132
    buf280 = buf279[0]
    buf281 = buf279[1]
    del buf279
    buf283 = buf282; del buf282  # reuse
    cpp_fused_native_batch_norm_backward_35(c_void_p(buf283.data_ptr()), c_void_p(squeeze_31.data_ptr()))
    del squeeze_31
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf285 = aten.convolution_backward(buf284, relu_3, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_131
    buf286 = buf285[0]
    buf287 = buf285[1]
    del buf285
    buf289 = buf276; del buf276  # reuse
    buf290 = buf274; del buf274  # reuse
    buf291 = empty((192, ), device='cpu', dtype=torch.float32)
    buf292 = empty((192, ), device='cpu', dtype=torch.float32)
    buf298 = empty((192, ), device='cpu', dtype=torch.float32)
    buf293 = empty((192, ), device='cpu', dtype=torch.float32)
    buf294 = buf284; del buf284  # reuse
    buf300 = buf278; del buf278  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf290.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(unsqueeze_858.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_870.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_882.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf300.data_ptr()))
    del buf280
    del buf286
    del buf288
    del buf290
    del buf292
    del convolution_6
    del convolution_7
    del primals_15
    del primals_17
    del primals_19
    del relu_3
    del squeeze_25
    del squeeze_28
    del unsqueeze_858
    del unsqueeze_870
    del unsqueeze_882
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf295 = aten.convolution_backward(buf294, relu_2, primals_130, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf294
    del primals_130
    buf296 = buf295[0]
    buf297 = buf295[1]
    del buf295
    buf299 = buf298; del buf298  # reuse
    cpp_fused_native_batch_norm_backward_37(c_void_p(buf299.data_ptr()), c_void_p(squeeze_22.data_ptr()))
    del squeeze_22
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf301 = aten.convolution_backward(buf300, relu_2, primals_129, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf300
    del primals_129
    buf302 = buf301[0]
    buf303 = buf301[1]
    del buf301
    buf304 = empty((96, ), device='cpu', dtype=torch.float32)
    buf305 = empty((96, ), device='cpu', dtype=torch.float32)
    buf311 = empty((96, ), device='cpu', dtype=torch.float32)
    buf317 = empty((96, ), device='cpu', dtype=torch.float32)
    buf306 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    buf312 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    buf318 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    buf307 = buf305; del buf305  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_38(c_void_p(buf307.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_894.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_906.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(unsqueeze_918.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf318.data_ptr()))
    del buf296
    del buf302
    del convolution_4
    del convolution_5
    del primals_11
    del primals_13
    del primals_9
    del relu_2
    del squeeze_19
    del unsqueeze_894
    del unsqueeze_906
    del unsqueeze_918
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf308 = aten.convolution_backward(buf306, relu_1, primals_128, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_128
    buf309 = buf308[0]
    buf310 = buf308[1]
    del buf308
    buf313 = buf311; del buf311  # reuse
    cpp_fused_native_batch_norm_backward_39(c_void_p(buf313.data_ptr()), c_void_p(squeeze_16.data_ptr()))
    del squeeze_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf314 = aten.convolution_backward(buf312, relu_1, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_127
    buf315 = buf314[0]
    buf316 = buf314[1]
    del buf314
    buf319 = buf317; del buf317  # reuse
    buf320 = empty((96, ), device='cpu', dtype=torch.float32)
    buf321 = empty((96, ), device='cpu', dtype=torch.float32)
    buf328 = empty((96, ), device='cpu', dtype=torch.float32)
    buf322 = buf312; del buf312  # reuse
    buf329 = buf306; del buf306  # reuse
    buf324 = buf322; del buf322  # reuse
    buf331 = buf329; del buf329  # reuse
    buf323 = buf321; del buf321  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_40(c_void_p(buf319.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_930.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_942.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf328.data_ptr()))
    del buf309
    del buf315
    del buf318
    del convolution_2
    del convolution_3
    del primals_5
    del primals_7
    del relu_1
    del squeeze_10
    del squeeze_13
    del unsqueeze_930
    del unsqueeze_942
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf325 = aten.convolution_backward(buf324, relu, primals_126, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf324
    del primals_126
    buf326 = buf325[0]
    buf327 = buf325[1]
    del buf325
    buf330 = buf328; del buf328  # reuse
    cpp_fused_native_batch_norm_backward_41(c_void_p(buf330.data_ptr()), c_void_p(squeeze_7.data_ptr()))
    del squeeze_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf332 = aten.convolution_backward(buf331, relu, primals_125, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf331
    del primals_125
    buf333 = buf332[0]
    buf334 = buf332[1]
    del buf332
    buf335 = empty((64, ), device='cpu', dtype=torch.float32)
    buf336 = empty((64, ), device='cpu', dtype=torch.float32)
    buf341 = empty((64, ), device='cpu', dtype=torch.float32)
    buf337 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    buf342 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    buf338 = buf336; del buf336  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_42(c_void_p(buf338.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_954.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_966.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf342.data_ptr()))
    del buf326
    del buf333
    del convolution
    del convolution_1
    del primals_1
    del primals_3
    del relu
    del squeeze_4
    del unsqueeze_954
    del unsqueeze_966
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf339 = aten.convolution_backward(buf337, primals_352, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf337
    del primals_124
    buf340 = buf339[1]
    del buf339
    buf343 = buf341; del buf341  # reuse
    cpp_fused_native_batch_norm_backward_43(c_void_p(buf343.data_ptr()), c_void_p(squeeze_1.data_ptr()))
    del squeeze_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf344 = aten.convolution_backward(buf342, primals_352, primals_123, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf342
    del primals_123
    del primals_352
    buf345 = buf344[1]
    return (buf343, buf335, buf338, buf335, buf330, buf320, buf323, buf320, buf319, buf304, buf313, buf304, buf307, buf304, buf299, buf291, buf293, buf291, buf289, buf275, buf283, buf275, buf277, buf275, buf273, buf256, buf266, buf256, buf259, buf256, buf255, buf240, buf249, buf240, buf243, buf240, buf235, buf227, buf229, buf227, buf225, buf211, buf219, buf211, buf213, buf211, buf209, buf195, buf203, buf195, buf197, buf195, buf193, buf179, buf187, buf179, buf181, buf179, buf177, buf163, buf171, buf163, buf165, buf163, buf161, buf147, buf155, buf147, buf149, buf147, buf145, buf131, buf139, buf131, buf133, buf131, buf129, buf115, buf123, buf115, buf117, buf115, buf113, buf99, buf107, buf99, buf101, buf99, buf97, buf83, buf91, buf83, buf85, buf83, buf81, buf67, buf75, buf67, buf69, buf67, buf65, buf51, buf59, buf51, buf53, buf51, buf49, buf32, buf42, buf32, buf35, buf32, buf31, buf16, buf25, buf16, buf19, buf16, buf11, buf3, buf5, buf3, buf345, buf340, buf334, buf327, buf316, buf310, buf303, buf297, buf287, buf281, buf270, buf263, buf252, buf246, buf239, buf233, buf223, buf217, buf207, buf201, buf191, buf185, buf175, buf169, buf159, buf153, buf143, buf137, buf127, buf121, buf111, buf105, buf95, buf89, buf79, buf73, buf63, buf57, buf46, buf39, buf28, buf22, buf15, buf9, reinterpret_tensor(buf1, (1000, 1408), (1408, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((64, 3, 1, 1), (3, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((96, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((192, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((192, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((1408, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1408, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 1408), (1408, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1408), (1408, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cpu', dtype=torch.bool)
    unsqueeze_246 = rand_strided((1, 1408, 1, 1), (1408, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1408, 1, 1), (1408, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_786 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_798 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_822 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_858 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_870 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_882 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_894 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_906 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_918 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_930 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_942 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_954 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_966 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_352, convolution, squeeze_1, convolution_1, squeeze_4, relu, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_1, squeeze_13, convolution_4, squeeze_16, convolution_5, squeeze_19, relu_2, convolution_6, squeeze_22, convolution_7, squeeze_25, relu_3, squeeze_28, convolution_8, squeeze_31, convolution_9, squeeze_34, relu_4, squeeze_37, convolution_10, squeeze_40, convolution_11, squeeze_43, relu_5, squeeze_46, convolution_12, squeeze_49, convolution_13, squeeze_52, relu_6, convolution_14, squeeze_55, convolution_15, squeeze_58, relu_7, squeeze_61, convolution_16, squeeze_64, convolution_17, squeeze_67, relu_8, squeeze_70, convolution_18, squeeze_73, convolution_19, squeeze_76, relu_9, squeeze_79, convolution_20, squeeze_82, convolution_21, squeeze_85, relu_10, squeeze_88, convolution_22, squeeze_91, convolution_23, squeeze_94, relu_11, squeeze_97, convolution_24, squeeze_100, convolution_25, squeeze_103, relu_12, squeeze_106, convolution_26, squeeze_109, convolution_27, squeeze_112, relu_13, squeeze_115, convolution_28, squeeze_118, convolution_29, squeeze_121, relu_14, squeeze_124, convolution_30, squeeze_127, convolution_31, squeeze_130, relu_15, squeeze_133, convolution_32, squeeze_136, convolution_33, squeeze_139, relu_16, squeeze_142, convolution_34, squeeze_145, convolution_35, squeeze_148, relu_17, squeeze_151, convolution_36, squeeze_154, convolution_37, squeeze_157, relu_18, squeeze_160, convolution_38, squeeze_163, convolution_39, squeeze_166, relu_19, squeeze_169, convolution_40, squeeze_172, convolution_41, squeeze_175, relu_20, convolution_42, squeeze_178, convolution_43, squeeze_181, clone, permute_1, le, unsqueeze_246, unsqueeze_258, unsqueeze_270, unsqueeze_282, unsqueeze_294, unsqueeze_306, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, unsqueeze_498, unsqueeze_510, unsqueeze_522, unsqueeze_534, unsqueeze_546, unsqueeze_558, unsqueeze_570, unsqueeze_582, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, unsqueeze_738, unsqueeze_750, unsqueeze_762, unsqueeze_774, unsqueeze_786, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, unsqueeze_858, unsqueeze_870, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('repvgg_a2', benchmark_compiled_module)
