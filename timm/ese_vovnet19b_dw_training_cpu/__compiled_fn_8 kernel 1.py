
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


cpp_fused_convolution_backward_div_hardsigmoid_backward_mul_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp5 = static_cast<float>(49.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = tmp10 / tmp6;
                            auto tmp12 = tmp9 + tmp11;
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = static_cast<float>(49.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = tmp10 / tmp6;
                        auto tmp12 = tmp9 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp16 = tmp14 - tmp15;
                        auto tmp18 = static_cast<float>(0.002551020408163265);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(out_ptr2 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
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
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1216L + x0 + (1440L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1216L + x1 + (1440L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (224L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
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
                tmp22.store(out_ptr3 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(992L + x0 + (1440L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(992L + x1 + (1440L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (224L*x0)));
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
                tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x0 + (1440L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x1 + (1440L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (224L*x0)));
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
                tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (224L*x0)));
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
                auto tmp10 = static_cast<float>(0.002551020408163265);
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
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1440L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (150528L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x2) + (150528L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = static_cast<float>(196.0);
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(196.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp15 = tmp13 - tmp14;
                        auto tmp17 = static_cast<float>(0.0006377551020408163);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = tmp20 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        auto tmp23 = tmp15 * tmp22;
                        auto tmp24 = tmp12 - tmp23;
                        auto tmp26 = tmp25 * tmp18;
                        auto tmp27 = tmp24 - tmp26;
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(896L + x0 + (1088L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(896L + x1 + (1088L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(704L + x0 + (1088L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(704L + x1 + (1088L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (1088L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1088L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1088L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x2) + (401408L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x2) + (401408L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x2) + (401408L*x1)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
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
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(576L + x0 + (736L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(576L + x1 + (736L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(416L + x0 + (736L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (160L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(416L + x1 + (736L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        #pragma omp single
        {
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
    }
}
''')


cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (736L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (160L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (736L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        #pragma omp single
        {
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
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (736L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_21 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x2) + (802816L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x2) + (802816L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x2) + (802816L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = static_cast<float>(3136.0);
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(3136.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp15 = tmp13 - tmp14;
                        auto tmp17 = static_cast<float>(3.985969387755102e-05);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = tmp20 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        auto tmp23 = tmp15 * tmp22;
                        auto tmp24 = tmp12 - tmp23;
                        auto tmp26 = tmp25 * tmp18;
                        auto tmp27 = tmp24 - tmp26;
                        tmp27.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(320L + x0 + (448L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(320L + x1 + (448L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x0 + (448L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1 + (448L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (448L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (448L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.cpp('''
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


cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_163, convolution, squeeze_1, relu, convolution_1, convolution_2, squeeze_4, relu_1, convolution_3, convolution_4, squeeze_7, relu_2, convolution_5, squeeze_10, relu_3, convolution_6, convolution_7, squeeze_13, relu_4, convolution_8, convolution_9, squeeze_16, relu_5, convolution_10, convolution_11, squeeze_19, cat, convolution_12, squeeze_22, relu_7, mean, div, mul_56, getitem_16, getitem_17, convolution_14, squeeze_25, relu_8, convolution_15, convolution_16, squeeze_28, relu_9, convolution_17, convolution_18, squeeze_31, relu_10, convolution_19, convolution_20, squeeze_34, cat_1, convolution_21, squeeze_37, relu_12, mean_1, div_1, mul_92, getitem_28, getitem_29, convolution_23, squeeze_40, relu_13, convolution_24, convolution_25, squeeze_43, relu_14, convolution_26, convolution_27, squeeze_46, relu_15, convolution_28, convolution_29, squeeze_49, cat_2, convolution_30, squeeze_52, relu_17, mean_2, div_2, mul_128, getitem_40, getitem_41, convolution_32, squeeze_55, relu_18, convolution_33, convolution_34, squeeze_58, relu_19, convolution_35, convolution_36, squeeze_61, relu_20, convolution_37, convolution_38, squeeze_64, cat_3, convolution_39, squeeze_67, relu_22, mean_3, div_3, clone, permute_1, bitwise_and, unsqueeze_94, le_1, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, bitwise_and_1, unsqueeze_154, le_6, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, bitwise_and_2, unsqueeze_214, le_11, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, bitwise_and_3, unsqueeze_274, le_16, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_17, (160, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (224, ), (1, ))
    assert_size_stride(primals_39, (224, ), (1, ))
    assert_size_stride(primals_41, (224, ), (1, ))
    assert_size_stride(primals_43, (224, ), (1, ))
    assert_size_stride(primals_45, (1024, ), (1, ))
    assert_size_stride(primals_47, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_48, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_49, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_50, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_52, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_53, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_54, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_55, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_57, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_58, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_59, (256, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_60, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_62, (160, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_64, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_65, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_66, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_67, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_69, (512, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_70, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_72, (192, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_73, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_75, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_76, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_77, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_78, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_79, (768, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_80, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_82, (224, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_83, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_84, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_85, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_86, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_87, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_88, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_89, (1024, 1440, 1, 1), (1440, 1, 1, 1))
    assert_size_stride(primals_90, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_163, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_1, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_2, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu_1, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_3, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_10, (128, ), (1, ))
    assert_size_stride(relu_3, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_6, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_7, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_13, (128, ), (1, ))
    assert_size_stride(relu_4, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_8, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_9, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_16, (128, ), (1, ))
    assert_size_stride(relu_5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_10, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_11, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_19, (128, ), (1, ))
    assert_size_stride(cat, (8, 448, 56, 56), (1404928, 1, 25088, 448))
    assert_size_stride(convolution_12, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_22, (256, ), (1, ))
    assert_size_stride(relu_7, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(mean, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(div, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_56, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(getitem_16, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_17, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_14, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(squeeze_25, (160, ), (1, ))
    assert_size_stride(relu_8, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_15, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_16, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(squeeze_28, (160, ), (1, ))
    assert_size_stride(relu_9, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_17, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_18, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(squeeze_31, (160, ), (1, ))
    assert_size_stride(relu_10, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_19, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_20, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(squeeze_34, (160, ), (1, ))
    assert_size_stride(cat_1, (8, 736, 28, 28), (577024, 1, 20608, 736))
    assert_size_stride(convolution_21, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_37, (512, ), (1, ))
    assert_size_stride(relu_12, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(mean_1, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(div_1, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(mul_92, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(getitem_28, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_29, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_23, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_40, (192, ), (1, ))
    assert_size_stride(relu_13, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_24, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_25, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_43, (192, ), (1, ))
    assert_size_stride(relu_14, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_26, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_27, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_46, (192, ), (1, ))
    assert_size_stride(relu_15, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_28, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_29, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_49, (192, ), (1, ))
    assert_size_stride(cat_2, (8, 1088, 14, 14), (213248, 1, 15232, 1088))
    assert_size_stride(convolution_30, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(squeeze_52, (768, ), (1, ))
    assert_size_stride(relu_17, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(mean_2, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(div_2, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(mul_128, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(getitem_40, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(getitem_41, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_32, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_55, (224, ), (1, ))
    assert_size_stride(relu_18, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_33, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_34, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_58, (224, ), (1, ))
    assert_size_stride(relu_19, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_35, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_36, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_61, (224, ), (1, ))
    assert_size_stride(relu_20, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_37, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_38, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(squeeze_64, (224, ), (1, ))
    assert_size_stride(cat_3, (8, 1440, 7, 7), (70560, 1, 10080, 1440))
    assert_size_stride(convolution_39, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_67, (1024, ), (1, ))
    assert_size_stride(relu_22, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(mean_3, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(div_3, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(clone, (8, 1024), (1024, 1))
    assert_size_stride(permute_1, (1000, 1024), (1024, 1))
    assert_size_stride(bitwise_and, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(unsqueeze_94, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_1, (8, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(unsqueeze_106, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(unsqueeze_118, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(unsqueeze_130, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(unsqueeze_142, (1, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(bitwise_and_1, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(unsqueeze_154, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(le_6, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(unsqueeze_166, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_190, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_202, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(bitwise_and_2, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(unsqueeze_214, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_11, (8, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(unsqueeze_226, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_238, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_and_3, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(unsqueeze_274, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_16, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(unsqueeze_286, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
    del clone
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cpu', dtype=torch.float32)
    buf4 = reinterpret_tensor(buf3, (8, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf3  # reuse
    cpp_fused_convolution_backward_div_hardsigmoid_backward_mul_sum_0(c_void_p(buf4.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(bitwise_and.data_ptr()), c_void_p(buf2.data_ptr()))
    del bitwise_and
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward]
    buf5 = aten.convolution_backward(buf4, mean_3, primals_90, [1024], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf4
    del mean_3
    del primals_90
    buf6 = buf5[0]
    buf7 = buf5[1]
    buf8 = buf5[2]
    del buf5
    buf9 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf10 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    buf12 = buf10; del buf10  # reuse
    buf13 = buf11; del buf11  # reuse
    cpp_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_1(c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_94.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf9.data_ptr()))
    del buf0
    del buf6
    del convolution_39
    del div_3
    del primals_45
    del relu_22
    del squeeze_67
    del unsqueeze_94
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf14 = aten.convolution_backward(buf13, cat_3, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf13
    del cat_3
    del primals_89
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty((224, ), device='cpu', dtype=torch.float32)
    buf18 = empty((224, ), device='cpu', dtype=torch.float32)
    buf19 = empty((224, ), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_2(c_void_p(le_1.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_106.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    del convolution_38
    del le_1
    del primals_43
    del squeeze_64
    del unsqueeze_106
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf21 = aten.convolution_backward(buf20, convolution_37, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf20
    del convolution_37
    del primals_88
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf24 = aten.convolution_backward(buf22, relu_20, primals_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False])
    del buf22
    del primals_87
    buf25 = buf24[0]
    buf26 = buf24[1]
    del buf24
    buf27 = buf18; del buf18  # reuse
    buf28 = empty((224, ), device='cpu', dtype=torch.float32)
    buf29 = buf25; del buf25  # reuse
    buf30 = buf28; del buf28  # reuse
    cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_3(c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_118.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf27.data_ptr()))
    del convolution_36
    del primals_41
    del relu_20
    del squeeze_61
    del unsqueeze_118
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf31 = aten.convolution_backward(buf29, convolution_35, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf29
    del convolution_35
    del primals_86
    buf32 = buf31[0]
    buf33 = buf31[1]
    del buf31
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf34 = aten.convolution_backward(buf32, relu_19, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False])
    del buf32
    del primals_85
    buf35 = buf34[0]
    buf36 = buf34[1]
    del buf34
    buf37 = empty((224, ), device='cpu', dtype=torch.float32)
    buf38 = empty((224, ), device='cpu', dtype=torch.float32)
    buf39 = buf35; del buf35  # reuse
    buf40 = buf38; del buf38  # reuse
    cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_130.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf37.data_ptr()))
    del convolution_34
    del primals_39
    del relu_19
    del squeeze_58
    del unsqueeze_130
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf41 = aten.convolution_backward(buf39, convolution_33, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf39
    del convolution_33
    del primals_84
    buf42 = buf41[0]
    buf43 = buf41[1]
    del buf41
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf44 = aten.convolution_backward(buf42, relu_18, primals_83, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 224, [True, True, False])
    del buf42
    del primals_83
    buf45 = buf44[0]
    buf46 = buf44[1]
    del buf44
    buf47 = empty((224, ), device='cpu', dtype=torch.float32)
    buf48 = empty((224, ), device='cpu', dtype=torch.float32)
    buf49 = empty((224, ), device='cpu', dtype=torch.float32)
    buf50 = buf45; del buf45  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(buf50.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_142.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    del buf48
    del convolution_32
    del primals_37
    del relu_18
    del squeeze_55
    del unsqueeze_142
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf51 = aten.convolution_backward(buf50, getitem_40, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf50
    del getitem_40
    del primals_82
    buf52 = buf51[0]
    buf53 = buf51[1]
    del buf51
    buf54 = buf52; del buf52  # reuse
    cpp_fused_add_6(c_void_p(buf54.data_ptr()), c_void_p(buf15.data_ptr()))
    del buf15
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf55 = aten.max_pool2d_with_indices_backward(buf54, mul_128, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_41)
    del getitem_41
    del mul_128
    buf56 = buf55
    del buf55
    buf57 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cpu', dtype=torch.float32)
    buf58 = reinterpret_tensor(buf57, (8, 768, 1, 1), (768, 1, 768, 768), 0); del buf57  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_7(c_void_p(buf58.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(bitwise_and_1.data_ptr()))
    del bitwise_and_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward]
    buf59 = aten.convolution_backward(buf58, mean_2, primals_80, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf58
    del mean_2
    del primals_80
    buf60 = buf59[0]
    buf61 = buf59[1]
    buf62 = buf59[2]
    del buf59
    buf63 = empty((768, ), device='cpu', dtype=torch.float32)
    buf64 = empty((768, ), device='cpu', dtype=torch.float32)
    buf65 = buf56; del buf56  # reuse
    buf66 = buf64; del buf64  # reuse
    buf67 = buf65; del buf65  # reuse
    cpp_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_8(c_void_p(buf67.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_154.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf63.data_ptr()))
    del buf60
    del convolution_30
    del div_2
    del primals_35
    del relu_17
    del squeeze_52
    del unsqueeze_154
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf68 = aten.convolution_backward(buf67, cat_2, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf67
    del cat_2
    del primals_79
    buf69 = buf68[0]
    buf70 = buf68[1]
    del buf68
    buf71 = empty((192, ), device='cpu', dtype=torch.float32)
    buf72 = empty((192, ), device='cpu', dtype=torch.float32)
    buf73 = empty((192, ), device='cpu', dtype=torch.float32)
    buf74 = reinterpret_tensor(buf54, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf54  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(le_6.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_166.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    del convolution_29
    del le_6
    del primals_33
    del squeeze_49
    del unsqueeze_166
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf75 = aten.convolution_backward(buf74, convolution_28, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf74
    del convolution_28
    del primals_78
    buf76 = buf75[0]
    buf77 = buf75[1]
    del buf75
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf78 = aten.convolution_backward(buf76, relu_15, primals_77, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
    del buf76
    del primals_77
    buf79 = buf78[0]
    buf80 = buf78[1]
    del buf78
    buf81 = buf72; del buf72  # reuse
    buf82 = empty((192, ), device='cpu', dtype=torch.float32)
    buf83 = buf79; del buf79  # reuse
    buf84 = buf82; del buf82  # reuse
    cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_178.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf81.data_ptr()))
    del convolution_27
    del primals_31
    del relu_15
    del squeeze_46
    del unsqueeze_178
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf85 = aten.convolution_backward(buf83, convolution_26, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf83
    del convolution_26
    del primals_76
    buf86 = buf85[0]
    buf87 = buf85[1]
    del buf85
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf88 = aten.convolution_backward(buf86, relu_14, primals_75, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
    del buf86
    del primals_75
    buf89 = buf88[0]
    buf90 = buf88[1]
    del buf88
    buf91 = empty((192, ), device='cpu', dtype=torch.float32)
    buf92 = empty((192, ), device='cpu', dtype=torch.float32)
    buf93 = buf89; del buf89  # reuse
    buf94 = buf92; del buf92  # reuse
    cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_190.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf91.data_ptr()))
    del convolution_25
    del primals_29
    del relu_14
    del squeeze_43
    del unsqueeze_190
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf95 = aten.convolution_backward(buf93, convolution_24, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf93
    del convolution_24
    del primals_74
    buf96 = buf95[0]
    buf97 = buf95[1]
    del buf95
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf98 = aten.convolution_backward(buf96, relu_13, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
    del buf96
    del primals_73
    buf99 = buf98[0]
    buf100 = buf98[1]
    del buf98
    buf101 = empty((192, ), device='cpu', dtype=torch.float32)
    buf102 = empty((192, ), device='cpu', dtype=torch.float32)
    buf103 = empty((192, ), device='cpu', dtype=torch.float32)
    buf104 = buf99; del buf99  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf104.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_202.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    del buf102
    del convolution_23
    del primals_27
    del relu_13
    del squeeze_40
    del unsqueeze_202
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf105 = aten.convolution_backward(buf104, getitem_28, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf104
    del getitem_28
    del primals_72
    buf106 = buf105[0]
    buf107 = buf105[1]
    del buf105
    buf108 = buf106; del buf106  # reuse
    cpp_fused_add_13(c_void_p(buf108.data_ptr()), c_void_p(buf69.data_ptr()))
    del buf69
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf109 = aten.max_pool2d_with_indices_backward(buf108, mul_92, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_29)
    del buf108
    del getitem_29
    del mul_92
    buf110 = buf109
    del buf109
    buf111 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf112 = reinterpret_tensor(buf111, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf111  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_14(c_void_p(buf112.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(bitwise_and_2.data_ptr()))
    del bitwise_and_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward]
    buf113 = aten.convolution_backward(buf112, mean_1, primals_70, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf112
    del mean_1
    del primals_70
    buf114 = buf113[0]
    buf115 = buf113[1]
    buf116 = buf113[2]
    del buf113
    buf117 = empty((512, ), device='cpu', dtype=torch.float32)
    buf118 = empty((512, ), device='cpu', dtype=torch.float32)
    buf119 = buf110; del buf110  # reuse
    buf120 = buf118; del buf118  # reuse
    buf121 = buf119; del buf119  # reuse
    cpp_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_15(c_void_p(buf121.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_214.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf117.data_ptr()))
    del buf114
    del convolution_21
    del div_1
    del primals_25
    del relu_12
    del squeeze_37
    del unsqueeze_214
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf122 = aten.convolution_backward(buf121, cat_1, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_1
    del primals_69
    buf123 = buf122[0]
    buf124 = buf122[1]
    del buf122
    buf125 = empty((160, ), device='cpu', dtype=torch.float32)
    buf126 = empty((160, ), device='cpu', dtype=torch.float32)
    buf127 = empty((160, ), device='cpu', dtype=torch.float32)
    buf128 = empty_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_16(c_void_p(le_11.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_226.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    del convolution_20
    del le_11
    del primals_23
    del squeeze_34
    del unsqueeze_226
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf129 = aten.convolution_backward(buf128, convolution_19, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf128
    del convolution_19
    del primals_68
    buf130 = buf129[0]
    buf131 = buf129[1]
    del buf129
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf132 = aten.convolution_backward(buf130, relu_10, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False])
    del buf130
    del primals_67
    buf133 = buf132[0]
    buf134 = buf132[1]
    del buf132
    buf135 = buf126; del buf126  # reuse
    buf136 = empty((160, ), device='cpu', dtype=torch.float32)
    buf137 = buf133; del buf133  # reuse
    buf138 = buf136; del buf136  # reuse
    cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_238.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf135.data_ptr()))
    del convolution_18
    del primals_21
    del relu_10
    del squeeze_31
    del unsqueeze_238
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf139 = aten.convolution_backward(buf137, convolution_17, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf137
    del convolution_17
    del primals_66
    buf140 = buf139[0]
    buf141 = buf139[1]
    del buf139
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf142 = aten.convolution_backward(buf140, relu_9, primals_65, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False])
    del buf140
    del primals_65
    buf143 = buf142[0]
    buf144 = buf142[1]
    del buf142
    buf145 = empty((160, ), device='cpu', dtype=torch.float32)
    buf146 = empty((160, ), device='cpu', dtype=torch.float32)
    buf147 = buf143; del buf143  # reuse
    buf148 = buf146; del buf146  # reuse
    cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_250.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf145.data_ptr()))
    del convolution_16
    del primals_19
    del relu_9
    del squeeze_28
    del unsqueeze_250
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf149 = aten.convolution_backward(buf147, convolution_15, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf147
    del convolution_15
    del primals_64
    buf150 = buf149[0]
    buf151 = buf149[1]
    del buf149
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf152 = aten.convolution_backward(buf150, relu_8, primals_63, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 160, [True, True, False])
    del buf150
    del primals_63
    buf153 = buf152[0]
    buf154 = buf152[1]
    del buf152
    buf155 = empty((160, ), device='cpu', dtype=torch.float32)
    buf156 = empty((160, ), device='cpu', dtype=torch.float32)
    buf157 = empty((160, ), device='cpu', dtype=torch.float32)
    buf158 = buf153; del buf153  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(buf158.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_262.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()))
    del buf156
    del convolution_14
    del primals_17
    del relu_8
    del squeeze_25
    del unsqueeze_262
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf159 = aten.convolution_backward(buf158, getitem_16, primals_62, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf158
    del getitem_16
    del primals_62
    buf160 = buf159[0]
    buf161 = buf159[1]
    del buf159
    buf162 = buf160; del buf160  # reuse
    cpp_fused_add_20(c_void_p(buf162.data_ptr()), c_void_p(buf123.data_ptr()))
    del buf123
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf163 = aten.max_pool2d_with_indices_backward(buf162, mul_56, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_17)
    del buf162
    del getitem_17
    del mul_56
    buf164 = buf163
    del buf163
    buf165 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf166 = reinterpret_tensor(buf165, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf165  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_21(c_void_p(buf166.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(bitwise_and_3.data_ptr()))
    del bitwise_and_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward]
    buf167 = aten.convolution_backward(buf166, mean, primals_60, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf166
    del mean
    del primals_60
    buf168 = buf167[0]
    buf169 = buf167[1]
    buf170 = buf167[2]
    del buf167
    buf171 = empty((256, ), device='cpu', dtype=torch.float32)
    buf172 = empty((256, ), device='cpu', dtype=torch.float32)
    buf173 = buf164; del buf164  # reuse
    buf174 = buf172; del buf172  # reuse
    buf175 = buf173; del buf173  # reuse
    cpp_fused_add_convolution_backward_div_hardsigmoid_backward_mul_native_batch_norm_backward_threshold_backward_22(c_void_p(buf175.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(div.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_274.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf171.data_ptr()))
    del buf168
    del convolution_12
    del div
    del primals_15
    del relu_7
    del squeeze_22
    del unsqueeze_274
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf176 = aten.convolution_backward(buf175, cat, primals_59, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf175
    del cat
    del primals_59
    buf177 = buf176[0]
    buf178 = buf176[1]
    del buf176
    buf179 = empty((128, ), device='cpu', dtype=torch.float32)
    buf180 = empty((128, ), device='cpu', dtype=torch.float32)
    buf181 = empty((128, ), device='cpu', dtype=torch.float32)
    buf182 = reinterpret_tensor(buf121, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf121  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(le_16.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_286.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()))
    del convolution_11
    del le_16
    del primals_13
    del squeeze_19
    del unsqueeze_286
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf183 = aten.convolution_backward(buf182, convolution_10, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf182
    del convolution_10
    del primals_58
    buf184 = buf183[0]
    buf185 = buf183[1]
    del buf183
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf186 = aten.convolution_backward(buf184, relu_5, primals_57, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf184
    del primals_57
    buf187 = buf186[0]
    buf188 = buf186[1]
    del buf186
    buf189 = buf180; del buf180  # reuse
    buf190 = empty((128, ), device='cpu', dtype=torch.float32)
    buf191 = buf187; del buf187  # reuse
    buf192 = buf190; del buf190  # reuse
    cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_298.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf189.data_ptr()))
    del convolution_9
    del primals_11
    del relu_5
    del squeeze_16
    del unsqueeze_298
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf193 = aten.convolution_backward(buf191, convolution_8, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf191
    del convolution_8
    del primals_56
    buf194 = buf193[0]
    buf195 = buf193[1]
    del buf193
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf196 = aten.convolution_backward(buf194, relu_4, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf194
    del primals_55
    buf197 = buf196[0]
    buf198 = buf196[1]
    del buf196
    buf199 = empty((128, ), device='cpu', dtype=torch.float32)
    buf200 = empty((128, ), device='cpu', dtype=torch.float32)
    buf201 = buf197; del buf197  # reuse
    buf202 = buf200; del buf200  # reuse
    cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_25(c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_310.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf199.data_ptr()))
    del convolution_7
    del primals_9
    del relu_4
    del squeeze_13
    del unsqueeze_310
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf203 = aten.convolution_backward(buf201, convolution_6, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf201
    del convolution_6
    del primals_54
    buf204 = buf203[0]
    buf205 = buf203[1]
    del buf203
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf206 = aten.convolution_backward(buf204, relu_3, primals_53, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf204
    del primals_53
    buf207 = buf206[0]
    buf208 = buf206[1]
    del buf206
    buf209 = empty((128, ), device='cpu', dtype=torch.float32)
    buf210 = empty((128, ), device='cpu', dtype=torch.float32)
    buf211 = empty((128, ), device='cpu', dtype=torch.float32)
    buf212 = buf207; del buf207  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_26(c_void_p(buf212.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del buf210
    del convolution_5
    del primals_7
    del relu_3
    del squeeze_10
    del unsqueeze_322
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf213 = aten.convolution_backward(buf212, relu_2, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf212
    del primals_52
    buf214 = buf213[0]
    buf215 = buf213[1]
    del buf213
    buf216 = empty((64, ), device='cpu', dtype=torch.float32)
    buf217 = empty((64, ), device='cpu', dtype=torch.float32)
    buf218 = buf214; del buf214  # reuse
    buf219 = buf217; del buf217  # reuse
    cpp_fused_add_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf216.data_ptr()))
    del buf177
    del convolution_4
    del primals_5
    del relu_2
    del squeeze_7
    del unsqueeze_334
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf220 = aten.convolution_backward(buf218, convolution_3, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf218
    del convolution_3
    del primals_51
    buf221 = buf220[0]
    buf222 = buf220[1]
    del buf220
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf223 = aten.convolution_backward(buf221, relu_1, primals_50, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
    del buf221
    del primals_50
    buf224 = buf223[0]
    buf225 = buf223[1]
    del buf223
    buf226 = empty((64, ), device='cpu', dtype=torch.float32)
    buf227 = empty((64, ), device='cpu', dtype=torch.float32)
    buf228 = empty((64, ), device='cpu', dtype=torch.float32)
    buf229 = buf224; del buf224  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_28(c_void_p(buf229.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del convolution_2
    del primals_3
    del relu_1
    del squeeze_4
    del unsqueeze_346
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf230 = aten.convolution_backward(buf229, convolution_1, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf229
    del convolution_1
    del primals_49
    buf231 = buf230[0]
    buf232 = buf230[1]
    del buf230
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf233 = aten.convolution_backward(buf231, relu, primals_48, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
    del buf231
    del primals_48
    buf234 = buf233[0]
    buf235 = buf233[1]
    del buf233
    buf236 = buf227; del buf227  # reuse
    buf237 = empty((64, ), device='cpu', dtype=torch.float32)
    buf238 = empty((64, ), device='cpu', dtype=torch.float32)
    buf239 = buf234; del buf234  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_native_batch_norm_backward_threshold_backward_29(c_void_p(buf239.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_358.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del buf237
    del convolution
    del primals_1
    del relu
    del squeeze_1
    del unsqueeze_358
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf240 = aten.convolution_backward(buf239, primals_163, primals_47, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf239
    del primals_163
    del primals_47
    buf241 = buf240[1]
    return (buf238, buf236, buf228, buf226, buf219, buf216, buf211, buf209, buf202, buf199, buf192, buf189, buf181, buf179, buf174, buf171, buf157, buf155, buf148, buf145, buf138, buf135, buf127, buf125, buf120, buf117, buf103, buf101, buf94, buf91, buf84, buf81, buf73, buf71, buf66, buf63, buf49, buf47, buf40, buf37, buf30, buf27, buf19, buf17, buf12, buf9, buf241, buf235, buf232, buf225, buf222, buf215, buf208, buf205, buf198, buf195, buf188, buf185, buf178, buf169, buf170, buf161, buf154, buf151, buf144, buf141, buf134, buf131, buf124, buf115, buf116, buf107, buf100, buf97, buf90, buf87, buf80, buf77, buf70, buf61, buf62, buf53, buf46, buf43, buf36, buf33, buf26, buf23, buf16, buf7, buf8, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((256, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((160, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((512, 736, 1, 1), (736, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((192, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, 1088, 1, 1), (1088, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((224, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((1024, 1440, 1, 1), (1440, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 448, 56, 56), (1404928, 1, 25088, 448), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    mul_56 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    getitem_16 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.int64)
    convolution_14 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 736, 28, 28), (577024, 1, 20608, 736), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    mul_92 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    getitem_28 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_29 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.int64)
    convolution_23 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 1088, 14, 14), (213248, 1, 15232, 1088), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    mul_128 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    getitem_40 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.int64)
    convolution_32 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((8, 1440, 7, 7), (70560, 1, 10080, 1440), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    bitwise_and = rand_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.bool)
    unsqueeze_94 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_1 = rand_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.bool)
    unsqueeze_106 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_118 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_142 = rand_strided((1, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.bool)
    unsqueeze_154 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_6 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.bool)
    unsqueeze_166 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_2 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.bool)
    unsqueeze_214 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_11 = rand_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.bool)
    unsqueeze_226 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_3 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.bool)
    unsqueeze_274 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_16 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.bool)
    unsqueeze_286 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_163, convolution, squeeze_1, relu, convolution_1, convolution_2, squeeze_4, relu_1, convolution_3, convolution_4, squeeze_7, relu_2, convolution_5, squeeze_10, relu_3, convolution_6, convolution_7, squeeze_13, relu_4, convolution_8, convolution_9, squeeze_16, relu_5, convolution_10, convolution_11, squeeze_19, cat, convolution_12, squeeze_22, relu_7, mean, div, mul_56, getitem_16, getitem_17, convolution_14, squeeze_25, relu_8, convolution_15, convolution_16, squeeze_28, relu_9, convolution_17, convolution_18, squeeze_31, relu_10, convolution_19, convolution_20, squeeze_34, cat_1, convolution_21, squeeze_37, relu_12, mean_1, div_1, mul_92, getitem_28, getitem_29, convolution_23, squeeze_40, relu_13, convolution_24, convolution_25, squeeze_43, relu_14, convolution_26, convolution_27, squeeze_46, relu_15, convolution_28, convolution_29, squeeze_49, cat_2, convolution_30, squeeze_52, relu_17, mean_2, div_2, mul_128, getitem_40, getitem_41, convolution_32, squeeze_55, relu_18, convolution_33, convolution_34, squeeze_58, relu_19, convolution_35, convolution_36, squeeze_61, relu_20, convolution_37, convolution_38, squeeze_64, cat_3, convolution_39, squeeze_67, relu_22, mean_3, div_3, clone, permute_1, bitwise_and, unsqueeze_94, le_1, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, bitwise_and_1, unsqueeze_154, le_6, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, bitwise_and_2, unsqueeze_214, le_11, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, bitwise_and_3, unsqueeze_274, le_16, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ese_vovnet19b_dw', benchmark_compiled_module)
