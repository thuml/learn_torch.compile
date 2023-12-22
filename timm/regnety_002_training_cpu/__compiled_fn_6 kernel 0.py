
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(49.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
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
                        tmp25.store(out_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.002551020408163265);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp4 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = static_cast<float>(49.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 / tmp7;
                            auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp4);
                            auto tmp11 = tmp9 + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp4 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = static_cast<float>(49.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 / tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp4);
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp15 = tmp13 - tmp14;
                        auto tmp17 = static_cast<float>(0.002551020408163265);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = tmp20 * tmp20;
                        auto tmp22 = tmp19 * tmp21;
                        auto tmp23 = tmp15 * tmp22;
                        auto tmp24 = tmp12 - tmp23;
                        auto tmp26 = tmp25 * tmp18;
                        auto tmp27 = tmp24 - tmp26;
                        tmp27.store(out_ptr2 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.002551020408163265);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp6 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = to_float_mask(tmp4 <= tmp2);
                        auto tmp8 = static_cast<float>(49.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = decltype(tmp2)::blendv(tmp10, tmp2, tmp6);
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp5);
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = decltype(tmp2)::blendv(tmp16, tmp2, tmp3);
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.002551020408163265);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.002551020408163265);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.002551020408163265);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.002551020408163265);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (368L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (368L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (368L*x2) + (18032L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2944L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (368L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (368L*x2) + (18032L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(368L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (368L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.002551020408163265);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (368L*x1) + (18032L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (368L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(368L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (368L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (368L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (152L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (152L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.0006377551020408163);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(238336L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (152L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.0006377551020408163);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (152L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (152L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.0006377551020408163);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(238336L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (152L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (152L*x2) + (29792L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (152L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (152L*x2) + (29792L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (152L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.0006377551020408163);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (152L*x1) + (29792L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (152L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (56L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (56L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (56L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (56L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (56L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (56L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (56L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (56L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (56L*x0)));
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
                    tmp25.store(out_ptr3 + static_cast<long>(x1 + (56L*x0)));
                    tmp39.store(out_ptr4 + static_cast<long>(x1 + (56L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (56L*x2) + (43904L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (56L*x2) + (43904L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (56L*x2) + (43904L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x2) + (43904L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (56L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (56L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (56L*x2) + (43904L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (56L*x1) + (43904L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (56L*x1) + (43904L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (56L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (56L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (56L*x1) + (43904L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(0.00015943877551020407);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (56L*x1) + (43904L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (56L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (56L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (56L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (56L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_63 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (24L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp25.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    tmp39.store(out_ptr4 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x2) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x2) + (75264L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_threshold_backward_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x2) + (75264L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x2) + (75264L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (24L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (24L*x2) + (75264L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (24L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (24L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
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
                        auto tmp18 = static_cast<float>(3.985969387755102e-05);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp23 = tmp20 * tmp22;
                        auto tmp24 = tmp16 * tmp23;
                        auto tmp25 = tmp13 - tmp24;
                        auto tmp27 = tmp26 * tmp19;
                        auto tmp28 = tmp25 - tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (24L*x1) + (75264L*x0)));
                    }
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_69 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_90, primals_91, primals_92, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_149, primals_150, primals_151, primals_152, primals_154, primals_156, primals_157, primals_158, primals_159, primals_161, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_319, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, mean, relu_3, convolution_4, mul_21, convolution_5, squeeze_10, convolution_6, squeeze_13, relu_4, convolution_7, squeeze_16, relu_5, convolution_8, squeeze_19, relu_6, mean_1, relu_7, convolution_10, mul_50, convolution_11, squeeze_22, convolution_12, squeeze_25, relu_8, convolution_13, squeeze_28, relu_9, convolution_14, squeeze_31, relu_10, mean_2, relu_11, convolution_16, mul_79, convolution_17, squeeze_34, convolution_18, squeeze_37, relu_12, convolution_19, squeeze_40, relu_13, convolution_20, squeeze_43, relu_14, mean_3, relu_15, convolution_22, mul_108, convolution_23, squeeze_46, relu_16, convolution_24, squeeze_49, relu_17, convolution_25, squeeze_52, relu_18, mean_4, relu_19, convolution_27, mul_130, convolution_28, squeeze_55, relu_20, convolution_29, squeeze_58, relu_21, convolution_30, squeeze_61, relu_22, mean_5, relu_23, convolution_32, mul_152, convolution_33, squeeze_64, relu_24, convolution_34, squeeze_67, relu_25, convolution_35, squeeze_70, relu_26, mean_6, relu_27, convolution_37, mul_174, convolution_38, squeeze_73, convolution_39, squeeze_76, relu_28, convolution_40, squeeze_79, relu_29, convolution_41, squeeze_82, relu_30, mean_7, relu_31, convolution_43, mul_203, convolution_44, squeeze_85, relu_32, convolution_45, squeeze_88, relu_33, convolution_46, squeeze_91, relu_34, mean_8, relu_35, convolution_48, mul_225, convolution_49, squeeze_94, relu_36, convolution_50, squeeze_97, relu_37, convolution_51, squeeze_100, relu_38, mean_9, relu_39, convolution_53, mul_247, convolution_54, squeeze_103, relu_40, convolution_55, squeeze_106, relu_41, convolution_56, squeeze_109, relu_42, mean_10, relu_43, convolution_58, mul_269, convolution_59, squeeze_112, relu_44, convolution_60, squeeze_115, relu_45, convolution_61, squeeze_118, relu_46, mean_11, relu_47, convolution_63, mul_291, convolution_64, squeeze_121, relu_48, convolution_65, squeeze_124, relu_49, convolution_66, squeeze_127, relu_50, mean_12, relu_51, convolution_68, mul_313, convolution_69, squeeze_130, clone, permute_1, le, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (24, ), (1, ))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_7, (24, ), (1, ))
    assert_size_stride(primals_9, (24, ), (1, ))
    assert_size_stride(primals_11, (56, ), (1, ))
    assert_size_stride(primals_13, (56, ), (1, ))
    assert_size_stride(primals_15, (56, ), (1, ))
    assert_size_stride(primals_17, (56, ), (1, ))
    assert_size_stride(primals_19, (152, ), (1, ))
    assert_size_stride(primals_21, (152, ), (1, ))
    assert_size_stride(primals_23, (152, ), (1, ))
    assert_size_stride(primals_25, (152, ), (1, ))
    assert_size_stride(primals_27, (152, ), (1, ))
    assert_size_stride(primals_29, (152, ), (1, ))
    assert_size_stride(primals_31, (152, ), (1, ))
    assert_size_stride(primals_33, (152, ), (1, ))
    assert_size_stride(primals_35, (152, ), (1, ))
    assert_size_stride(primals_37, (152, ), (1, ))
    assert_size_stride(primals_39, (152, ), (1, ))
    assert_size_stride(primals_41, (152, ), (1, ))
    assert_size_stride(primals_43, (152, ), (1, ))
    assert_size_stride(primals_45, (368, ), (1, ))
    assert_size_stride(primals_47, (368, ), (1, ))
    assert_size_stride(primals_49, (368, ), (1, ))
    assert_size_stride(primals_51, (368, ), (1, ))
    assert_size_stride(primals_53, (368, ), (1, ))
    assert_size_stride(primals_55, (368, ), (1, ))
    assert_size_stride(primals_57, (368, ), (1, ))
    assert_size_stride(primals_59, (368, ), (1, ))
    assert_size_stride(primals_61, (368, ), (1, ))
    assert_size_stride(primals_63, (368, ), (1, ))
    assert_size_stride(primals_65, (368, ), (1, ))
    assert_size_stride(primals_67, (368, ), (1, ))
    assert_size_stride(primals_69, (368, ), (1, ))
    assert_size_stride(primals_71, (368, ), (1, ))
    assert_size_stride(primals_73, (368, ), (1, ))
    assert_size_stride(primals_75, (368, ), (1, ))
    assert_size_stride(primals_77, (368, ), (1, ))
    assert_size_stride(primals_79, (368, ), (1, ))
    assert_size_stride(primals_81, (368, ), (1, ))
    assert_size_stride(primals_83, (368, ), (1, ))
    assert_size_stride(primals_85, (368, ), (1, ))
    assert_size_stride(primals_87, (368, ), (1, ))
    assert_size_stride(primals_89, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_90, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_91, (24, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_92, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_94, (24, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_96, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_97, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_98, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_99, (56, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_100, (6, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_102, (56, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_104, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_105, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_106, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_107, (152, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_108, (14, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_110, (152, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(primals_112, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_113, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_114, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_115, (152, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_116, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_118, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_120, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_121, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_122, (152, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_123, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_125, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_127, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_128, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_129, (152, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_130, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_132, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_134, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_135, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_136, (368, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_137, (38, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_139, (368, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_141, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_142, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_143, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_144, (368, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_145, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_147, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_149, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_150, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_151, (368, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_152, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_154, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_156, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_157, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_158, (368, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_159, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_161, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_163, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_164, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_165, (368, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_166, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_168, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_170, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_171, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_172, (368, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_173, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_175, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_177, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_178, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_179, (368, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_180, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_182, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_184, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_319, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(squeeze_4, (24, ), (1, ))
    assert_size_stride(relu_1, (8, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(convolution_2, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_7, (24, ), (1, ))
    assert_size_stride(relu_2, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(mean, (8, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(relu_3, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_4, (8, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(mul_21, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_5, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_10, (24, ), (1, ))
    assert_size_stride(convolution_6, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_13, (24, ), (1, ))
    assert_size_stride(relu_4, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_7, (8, 56, 56, 56), (175616, 1, 3136, 56))
    assert_size_stride(squeeze_16, (56, ), (1, ))
    assert_size_stride(relu_5, (8, 56, 56, 56), (175616, 1, 3136, 56))
    assert_size_stride(convolution_8, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_19, (56, ), (1, ))
    assert_size_stride(relu_6, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(mean_1, (8, 56, 1, 1), (56, 1, 56, 56))
    assert_size_stride(relu_7, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_10, (8, 56, 1, 1), (56, 1, 56, 56))
    assert_size_stride(mul_50, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(convolution_11, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_22, (56, ), (1, ))
    assert_size_stride(convolution_12, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_25, (56, ), (1, ))
    assert_size_stride(relu_8, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(convolution_13, (8, 152, 28, 28), (119168, 1, 4256, 152))
    assert_size_stride(squeeze_28, (152, ), (1, ))
    assert_size_stride(relu_9, (8, 152, 28, 28), (119168, 1, 4256, 152))
    assert_size_stride(convolution_14, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_31, (152, ), (1, ))
    assert_size_stride(relu_10, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(mean_2, (8, 152, 1, 1), (152, 1, 152, 152))
    assert_size_stride(relu_11, (8, 14, 1, 1), (14, 1, 14, 14))
    assert_size_stride(convolution_16, (8, 152, 1, 1), (152, 1, 152, 152))
    assert_size_stride(mul_79, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_17, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_34, (152, ), (1, ))
    assert_size_stride(convolution_18, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_37, (152, ), (1, ))
    assert_size_stride(relu_12, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_19, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_40, (152, ), (1, ))
    assert_size_stride(relu_13, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_20, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_43, (152, ), (1, ))
    assert_size_stride(relu_14, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(mean_3, (8, 152, 1, 1), (152, 1, 152, 152))
    assert_size_stride(relu_15, (8, 38, 1, 1), (38, 1, 38, 38))
    assert_size_stride(convolution_22, (8, 152, 1, 1), (152, 1, 152, 152))
    assert_size_stride(mul_108, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_23, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_46, (152, ), (1, ))
    assert_size_stride(relu_16, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_24, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_49, (152, ), (1, ))
    assert_size_stride(relu_17, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_25, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_52, (152, ), (1, ))
    assert_size_stride(relu_18, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(mean_4, (8, 152, 1, 1), (152, 1, 152, 152))
    assert_size_stride(relu_19, (8, 38, 1, 1), (38, 1, 38, 38))
    assert_size_stride(convolution_27, (8, 152, 1, 1), (152, 1, 152, 152))
    assert_size_stride(mul_130, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_28, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_55, (152, ), (1, ))
    assert_size_stride(relu_20, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_29, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_58, (152, ), (1, ))
    assert_size_stride(relu_21, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_30, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_61, (152, ), (1, ))
    assert_size_stride(relu_22, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(mean_5, (8, 152, 1, 1), (152, 1, 152, 152))
    assert_size_stride(relu_23, (8, 38, 1, 1), (38, 1, 38, 38))
    assert_size_stride(convolution_32, (8, 152, 1, 1), (152, 1, 152, 152))
    assert_size_stride(mul_152, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_33, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(squeeze_64, (152, ), (1, ))
    assert_size_stride(relu_24, (8, 152, 14, 14), (29792, 1, 2128, 152))
    assert_size_stride(convolution_34, (8, 368, 14, 14), (72128, 1, 5152, 368))
    assert_size_stride(squeeze_67, (368, ), (1, ))
    assert_size_stride(relu_25, (8, 368, 14, 14), (72128, 1, 5152, 368))
    assert_size_stride(convolution_35, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_70, (368, ), (1, ))
    assert_size_stride(relu_26, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(mean_6, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(relu_27, (8, 38, 1, 1), (38, 1, 38, 38))
    assert_size_stride(convolution_37, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(mul_174, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_38, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_73, (368, ), (1, ))
    assert_size_stride(convolution_39, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_76, (368, ), (1, ))
    assert_size_stride(relu_28, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_40, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_79, (368, ), (1, ))
    assert_size_stride(relu_29, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_41, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_82, (368, ), (1, ))
    assert_size_stride(relu_30, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(mean_7, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(relu_31, (8, 92, 1, 1), (92, 1, 92, 92))
    assert_size_stride(convolution_43, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(mul_203, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_44, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_85, (368, ), (1, ))
    assert_size_stride(relu_32, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_45, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_88, (368, ), (1, ))
    assert_size_stride(relu_33, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_46, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_91, (368, ), (1, ))
    assert_size_stride(relu_34, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(mean_8, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(relu_35, (8, 92, 1, 1), (92, 1, 92, 92))
    assert_size_stride(convolution_48, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(mul_225, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_49, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_94, (368, ), (1, ))
    assert_size_stride(relu_36, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_50, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_97, (368, ), (1, ))
    assert_size_stride(relu_37, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_51, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_100, (368, ), (1, ))
    assert_size_stride(relu_38, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(mean_9, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(relu_39, (8, 92, 1, 1), (92, 1, 92, 92))
    assert_size_stride(convolution_53, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(mul_247, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_54, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_103, (368, ), (1, ))
    assert_size_stride(relu_40, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_55, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_106, (368, ), (1, ))
    assert_size_stride(relu_41, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_56, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_109, (368, ), (1, ))
    assert_size_stride(relu_42, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(mean_10, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(relu_43, (8, 92, 1, 1), (92, 1, 92, 92))
    assert_size_stride(convolution_58, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(mul_269, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_59, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_112, (368, ), (1, ))
    assert_size_stride(relu_44, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_60, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_115, (368, ), (1, ))
    assert_size_stride(relu_45, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_61, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_118, (368, ), (1, ))
    assert_size_stride(relu_46, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(mean_11, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(relu_47, (8, 92, 1, 1), (92, 1, 92, 92))
    assert_size_stride(convolution_63, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(mul_291, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_64, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_121, (368, ), (1, ))
    assert_size_stride(relu_48, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_65, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_124, (368, ), (1, ))
    assert_size_stride(relu_49, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_66, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_127, (368, ), (1, ))
    assert_size_stride(relu_50, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(mean_12, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(relu_51, (8, 92, 1, 1), (92, 1, 92, 92))
    assert_size_stride(convolution_68, (8, 368, 1, 1), (368, 1, 368, 368))
    assert_size_stride(mul_313, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(convolution_69, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(squeeze_130, (368, ), (1, ))
    assert_size_stride(clone, (8, 368), (368, 1))
    assert_size_stride(permute_1, (1000, 368), (368, 1))
    assert_size_stride(le, (8, 368, 7, 7), (18032, 1, 2576, 368))
    assert_size_stride(unsqueeze_178, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_190, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_202, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_214, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_238, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_514, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_586, (1, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_682, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 368), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 368), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
    del clone
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((368, ), device='cpu', dtype=torch.float32)
    buf4 = empty((368, ), device='cpu', dtype=torch.float32)
    buf5 = empty((368, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(unsqueeze_178.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del convolution_69
    del primals_87
    del squeeze_130
    del tangents_1
    del unsqueeze_178
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, mul_313, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf6
    del mul_313
    del primals_184
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cpu', dtype=torch.float32)
    buf11 = reinterpret_tensor(buf10, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf10  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_1(c_void_p(buf11.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(relu_50.data_ptr()), c_void_p(convolution_68.data_ptr()))
    # Source Nodes: [sigmoid_12], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf12 = aten.convolution_backward(buf11, relu_51, primals_182, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf11
    del primals_182
    buf13 = buf12[0]
    buf14 = buf12[1]
    buf15 = buf12[2]
    del buf12
    buf16 = buf13; del buf13  # reuse
    cpp_fused_convolution_backward_threshold_backward_2(c_void_p(buf16.data_ptr()), c_void_p(relu_51.data_ptr()))
    del relu_51
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf17 = aten.convolution_backward(buf16, mean_12, primals_180, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf16
    del mean_12
    del primals_180
    buf18 = buf17[0]
    buf19 = buf17[1]
    buf20 = buf17[2]
    del buf17
    buf21 = buf4; del buf4  # reuse
    buf22 = empty((368, ), device='cpu', dtype=torch.float32)
    buf23 = buf8; del buf8  # reuse
    buf24 = buf22; del buf22  # reuse
    buf25 = buf23; del buf23  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_3(c_void_p(buf25.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(relu_50.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(unsqueeze_190.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf21.data_ptr()))
    del convolution_66
    del convolution_68
    del primals_85
    del relu_50
    del squeeze_127
    del unsqueeze_190
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf26 = aten.convolution_backward(buf25, relu_49, primals_179, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
    del buf25
    del primals_179
    buf27 = buf26[0]
    buf28 = buf26[1]
    del buf26
    buf29 = empty((368, ), device='cpu', dtype=torch.float32)
    buf30 = empty((368, ), device='cpu', dtype=torch.float32)
    buf31 = empty((368, ), device='cpu', dtype=torch.float32)
    buf32 = buf27; del buf27  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf32.data_ptr()), c_void_p(relu_49.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_202.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del convolution_65
    del primals_83
    del relu_49
    del squeeze_124
    del unsqueeze_202
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf33 = aten.convolution_backward(buf32, relu_48, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_178
    buf34 = buf33[0]
    buf35 = buf33[1]
    del buf33
    buf36 = buf30; del buf30  # reuse
    buf37 = empty((368, ), device='cpu', dtype=torch.float32)
    buf38 = buf32; del buf32  # reuse
    buf39 = buf37; del buf37  # reuse
    buf40 = buf38; del buf38  # reuse
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_5(c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(relu_48.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_214.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf36.data_ptr()))
    del convolution_64
    del primals_81
    del squeeze_121
    del unsqueeze_214
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf41 = aten.convolution_backward(buf40, mul_291, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf40
    del mul_291
    del primals_177
    buf42 = buf41[0]
    buf43 = buf41[1]
    del buf41
    buf44 = reinterpret_tensor(buf18, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf18  # reuse
    buf45 = reinterpret_tensor(buf44, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf44  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_6(c_void_p(buf45.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(convolution_63.data_ptr()))
    # Source Nodes: [sigmoid_11], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf46 = aten.convolution_backward(buf45, relu_47, primals_175, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf45
    del primals_175
    buf47 = buf46[0]
    buf48 = buf46[1]
    buf49 = buf46[2]
    del buf46
    buf50 = buf47; del buf47  # reuse
    cpp_fused_convolution_backward_threshold_backward_7(c_void_p(buf50.data_ptr()), c_void_p(relu_47.data_ptr()))
    del relu_47
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf51 = aten.convolution_backward(buf50, mean_11, primals_173, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf50
    del mean_11
    del primals_173
    buf52 = buf51[0]
    buf53 = buf51[1]
    buf54 = buf51[2]
    del buf51
    buf55 = empty((368, ), device='cpu', dtype=torch.float32)
    buf56 = empty((368, ), device='cpu', dtype=torch.float32)
    buf57 = buf42; del buf42  # reuse
    buf58 = buf56; del buf56  # reuse
    buf59 = buf57; del buf57  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_8(c_void_p(buf59.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_226.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf55.data_ptr()))
    del buf52
    del convolution_61
    del convolution_63
    del primals_79
    del relu_46
    del squeeze_118
    del unsqueeze_226
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf60 = aten.convolution_backward(buf59, relu_45, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
    del buf59
    del primals_172
    buf61 = buf60[0]
    buf62 = buf60[1]
    del buf60
    buf63 = empty((368, ), device='cpu', dtype=torch.float32)
    buf64 = empty((368, ), device='cpu', dtype=torch.float32)
    buf65 = empty((368, ), device='cpu', dtype=torch.float32)
    buf66 = buf61; del buf61  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf66.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_238.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()))
    del convolution_60
    del primals_77
    del relu_45
    del squeeze_115
    del unsqueeze_238
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf67 = aten.convolution_backward(buf66, relu_44, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_171
    buf68 = buf67[0]
    buf69 = buf67[1]
    del buf67
    buf70 = buf34; del buf34  # reuse
    buf71 = buf64; del buf64  # reuse
    buf72 = empty((368, ), device='cpu', dtype=torch.float32)
    buf73 = empty((368, ), device='cpu', dtype=torch.float32)
    buf74 = buf66; del buf66  # reuse
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_10(c_void_p(buf70.data_ptr()), c_void_p(relu_44.data_ptr()), c_void_p(relu_48.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_250.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    del buf68
    del convolution_59
    del le
    del primals_75
    del relu_44
    del relu_48
    del squeeze_112
    del unsqueeze_250
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf75 = aten.convolution_backward(buf74, mul_269, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf74
    del mul_269
    del primals_170
    buf76 = buf75[0]
    buf77 = buf75[1]
    del buf75
    buf78 = reinterpret_tensor(buf0, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf0  # reuse
    buf79 = reinterpret_tensor(buf78, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf78  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_11(c_void_p(buf79.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(convolution_58.data_ptr()))
    # Source Nodes: [sigmoid_10], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf80 = aten.convolution_backward(buf79, relu_43, primals_168, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf79
    del primals_168
    buf81 = buf80[0]
    buf82 = buf80[1]
    buf83 = buf80[2]
    del buf80
    buf84 = buf81; del buf81  # reuse
    cpp_fused_convolution_backward_threshold_backward_12(c_void_p(buf84.data_ptr()), c_void_p(relu_43.data_ptr()))
    del relu_43
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf85 = aten.convolution_backward(buf84, mean_10, primals_166, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf84
    del mean_10
    del primals_166
    buf86 = buf85[0]
    buf87 = buf85[1]
    buf88 = buf85[2]
    del buf85
    buf89 = buf72; del buf72  # reuse
    buf90 = empty((368, ), device='cpu', dtype=torch.float32)
    buf91 = buf76; del buf76  # reuse
    buf92 = buf90; del buf90  # reuse
    buf93 = buf91; del buf91  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_13(c_void_p(buf93.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_262.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf89.data_ptr()))
    del convolution_56
    del convolution_58
    del primals_73
    del relu_42
    del squeeze_109
    del unsqueeze_262
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf94 = aten.convolution_backward(buf93, relu_41, primals_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
    del buf93
    del primals_165
    buf95 = buf94[0]
    buf96 = buf94[1]
    del buf94
    buf97 = empty((368, ), device='cpu', dtype=torch.float32)
    buf98 = empty((368, ), device='cpu', dtype=torch.float32)
    buf99 = empty((368, ), device='cpu', dtype=torch.float32)
    buf100 = buf95; del buf95  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf100.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_274.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    del convolution_55
    del primals_71
    del relu_41
    del squeeze_106
    del unsqueeze_274
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf101 = aten.convolution_backward(buf100, relu_40, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_164
    buf102 = buf101[0]
    buf103 = buf101[1]
    del buf101
    buf104 = buf98; del buf98  # reuse
    buf105 = empty((368, ), device='cpu', dtype=torch.float32)
    buf106 = buf100; del buf100  # reuse
    buf107 = buf105; del buf105  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_15(c_void_p(buf107.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_286.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()))
    del convolution_54
    del primals_69
    del squeeze_103
    del unsqueeze_286
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf108 = aten.convolution_backward(buf106, mul_247, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf106
    del mul_247
    del primals_163
    buf109 = buf108[0]
    buf110 = buf108[1]
    del buf108
    buf111 = reinterpret_tensor(buf86, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf86  # reuse
    buf112 = reinterpret_tensor(buf111, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf111  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_16(c_void_p(buf112.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_53.data_ptr()))
    # Source Nodes: [sigmoid_9], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf113 = aten.convolution_backward(buf112, relu_39, primals_161, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf112
    del primals_161
    buf114 = buf113[0]
    buf115 = buf113[1]
    buf116 = buf113[2]
    del buf113
    buf117 = buf114; del buf114  # reuse
    cpp_fused_convolution_backward_threshold_backward_17(c_void_p(buf117.data_ptr()), c_void_p(relu_39.data_ptr()))
    del relu_39
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf118 = aten.convolution_backward(buf117, mean_9, primals_159, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf117
    del mean_9
    del primals_159
    buf119 = buf118[0]
    buf120 = buf118[1]
    buf121 = buf118[2]
    del buf118
    buf122 = empty((368, ), device='cpu', dtype=torch.float32)
    buf123 = empty((368, ), device='cpu', dtype=torch.float32)
    buf124 = buf109; del buf109  # reuse
    buf125 = buf123; del buf123  # reuse
    buf126 = buf124; del buf124  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_18(c_void_p(buf126.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_298.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf122.data_ptr()))
    del convolution_51
    del convolution_53
    del primals_67
    del relu_38
    del squeeze_100
    del unsqueeze_298
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf127 = aten.convolution_backward(buf126, relu_37, primals_158, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
    del buf126
    del primals_158
    buf128 = buf127[0]
    buf129 = buf127[1]
    del buf127
    buf130 = empty((368, ), device='cpu', dtype=torch.float32)
    buf131 = empty((368, ), device='cpu', dtype=torch.float32)
    buf132 = empty((368, ), device='cpu', dtype=torch.float32)
    buf133 = buf128; del buf128  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(buf133.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_310.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del convolution_50
    del primals_65
    del relu_37
    del squeeze_97
    del unsqueeze_310
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf134 = aten.convolution_backward(buf133, relu_36, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_157
    buf135 = buf134[0]
    buf136 = buf134[1]
    del buf134
    buf137 = buf102; del buf102  # reuse
    buf138 = buf131; del buf131  # reuse
    buf139 = empty((368, ), device='cpu', dtype=torch.float32)
    buf140 = empty((368, ), device='cpu', dtype=torch.float32)
    buf141 = buf133; del buf133  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(buf137.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    del buf135
    del buf70
    del convolution_49
    del primals_63
    del relu_36
    del relu_40
    del squeeze_94
    del unsqueeze_322
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf142 = aten.convolution_backward(buf141, mul_225, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf141
    del mul_225
    del primals_156
    buf143 = buf142[0]
    buf144 = buf142[1]
    del buf142
    buf145 = reinterpret_tensor(buf119, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf119  # reuse
    buf146 = reinterpret_tensor(buf145, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf145  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_21(c_void_p(buf146.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_48.data_ptr()))
    # Source Nodes: [sigmoid_8], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf147 = aten.convolution_backward(buf146, relu_35, primals_154, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf146
    del primals_154
    buf148 = buf147[0]
    buf149 = buf147[1]
    buf150 = buf147[2]
    del buf147
    buf151 = buf148; del buf148  # reuse
    cpp_fused_convolution_backward_threshold_backward_22(c_void_p(buf151.data_ptr()), c_void_p(relu_35.data_ptr()))
    del relu_35
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf152 = aten.convolution_backward(buf151, mean_8, primals_152, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf151
    del mean_8
    del primals_152
    buf153 = buf152[0]
    buf154 = buf152[1]
    buf155 = buf152[2]
    del buf152
    buf156 = buf139; del buf139  # reuse
    buf157 = empty((368, ), device='cpu', dtype=torch.float32)
    buf158 = buf143; del buf143  # reuse
    buf159 = buf157; del buf157  # reuse
    buf160 = buf158; del buf158  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_23(c_void_p(buf160.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf156.data_ptr()))
    del convolution_46
    del convolution_48
    del primals_61
    del relu_34
    del squeeze_91
    del unsqueeze_334
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf161 = aten.convolution_backward(buf160, relu_33, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
    del buf160
    del primals_151
    buf162 = buf161[0]
    buf163 = buf161[1]
    del buf161
    buf164 = empty((368, ), device='cpu', dtype=torch.float32)
    buf165 = empty((368, ), device='cpu', dtype=torch.float32)
    buf166 = empty((368, ), device='cpu', dtype=torch.float32)
    buf167 = buf162; del buf162  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf167.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    del convolution_45
    del primals_59
    del relu_33
    del squeeze_88
    del unsqueeze_346
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf168 = aten.convolution_backward(buf167, relu_32, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_150
    buf169 = buf168[0]
    buf170 = buf168[1]
    del buf168
    buf171 = buf165; del buf165  # reuse
    buf172 = empty((368, ), device='cpu', dtype=torch.float32)
    buf173 = buf167; del buf167  # reuse
    buf174 = buf172; del buf172  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_25(c_void_p(buf174.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_358.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()))
    del convolution_44
    del primals_57
    del squeeze_85
    del unsqueeze_358
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf175 = aten.convolution_backward(buf173, mul_203, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf173
    del mul_203
    del primals_149
    buf176 = buf175[0]
    buf177 = buf175[1]
    del buf175
    buf178 = reinterpret_tensor(buf153, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf153  # reuse
    buf179 = reinterpret_tensor(buf178, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf178  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_26(c_void_p(buf179.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(convolution_43.data_ptr()))
    # Source Nodes: [sigmoid_7], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf180 = aten.convolution_backward(buf179, relu_31, primals_147, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf179
    del primals_147
    buf181 = buf180[0]
    buf182 = buf180[1]
    buf183 = buf180[2]
    del buf180
    buf184 = buf181; del buf181  # reuse
    cpp_fused_convolution_backward_threshold_backward_27(c_void_p(buf184.data_ptr()), c_void_p(relu_31.data_ptr()))
    del relu_31
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf185 = aten.convolution_backward(buf184, mean_7, primals_145, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf184
    del mean_7
    del primals_145
    buf186 = buf185[0]
    buf187 = buf185[1]
    buf188 = buf185[2]
    del buf185
    buf189 = empty((368, ), device='cpu', dtype=torch.float32)
    buf190 = empty((368, ), device='cpu', dtype=torch.float32)
    buf191 = buf176; del buf176  # reuse
    buf192 = buf190; del buf190  # reuse
    buf193 = buf191; del buf191  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_28(c_void_p(buf193.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_370.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf189.data_ptr()))
    del convolution_41
    del convolution_43
    del primals_55
    del relu_30
    del squeeze_82
    del unsqueeze_370
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf194 = aten.convolution_backward(buf193, relu_29, primals_144, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
    del primals_144
    buf195 = buf194[0]
    buf196 = buf194[1]
    del buf194
    buf197 = empty((368, ), device='cpu', dtype=torch.float32)
    buf198 = empty((368, ), device='cpu', dtype=torch.float32)
    buf199 = empty((368, ), device='cpu', dtype=torch.float32)
    buf200 = buf195; del buf195  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29(c_void_p(buf200.data_ptr()), c_void_p(relu_29.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_382.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    del convolution_40
    del primals_53
    del relu_29
    del squeeze_79
    del unsqueeze_382
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf201 = aten.convolution_backward(buf200, relu_28, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_143
    buf202 = buf201[0]
    buf203 = buf201[1]
    del buf201
    buf204 = buf137; del buf137  # reuse
    buf205 = buf198; del buf198  # reuse
    buf206 = empty((368, ), device='cpu', dtype=torch.float32)
    buf212 = empty((368, ), device='cpu', dtype=torch.float32)
    buf207 = empty((368, ), device='cpu', dtype=torch.float32)
    buf208 = buf200; del buf200  # reuse
    buf214 = buf193; del buf193  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_30(c_void_p(buf204.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_406.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf214.data_ptr()))
    del buf169
    del buf202
    del buf204
    del convolution_38
    del convolution_39
    del primals_49
    del primals_51
    del relu_28
    del relu_32
    del squeeze_76
    del unsqueeze_394
    del unsqueeze_406
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf209 = aten.convolution_backward(buf208, relu_24, primals_142, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf208
    del primals_142
    buf210 = buf209[0]
    buf211 = buf209[1]
    del buf209
    buf213 = buf212; del buf212  # reuse
    cpp_fused_native_batch_norm_backward_31(c_void_p(buf213.data_ptr()), c_void_p(squeeze_73.data_ptr()))
    del squeeze_73
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf215 = aten.convolution_backward(buf214, mul_174, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf214
    del mul_174
    del primals_141
    buf216 = buf215[0]
    buf217 = buf215[1]
    del buf215
    buf218 = reinterpret_tensor(buf186, (8, 368, 1, 1), (368, 1, 2944, 2944), 0); del buf186  # reuse
    buf219 = reinterpret_tensor(buf218, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf218  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_32(c_void_p(buf219.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_37.data_ptr()))
    # Source Nodes: [sigmoid_6], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf220 = aten.convolution_backward(buf219, relu_27, primals_139, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf219
    del primals_139
    buf221 = buf220[0]
    buf222 = buf220[1]
    buf223 = buf220[2]
    del buf220
    buf224 = buf221; del buf221  # reuse
    cpp_fused_convolution_backward_threshold_backward_33(c_void_p(buf224.data_ptr()), c_void_p(relu_27.data_ptr()))
    del relu_27
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf225 = aten.convolution_backward(buf224, mean_6, primals_137, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf224
    del mean_6
    del primals_137
    buf226 = buf225[0]
    buf227 = buf225[1]
    buf228 = buf225[2]
    del buf225
    buf229 = buf206; del buf206  # reuse
    buf230 = empty((368, ), device='cpu', dtype=torch.float32)
    buf231 = buf216; del buf216  # reuse
    buf232 = buf230; del buf230  # reuse
    buf233 = buf231; del buf231  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_34(c_void_p(buf233.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_418.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf229.data_ptr()))
    del buf226
    del convolution_35
    del convolution_37
    del primals_47
    del relu_26
    del squeeze_70
    del unsqueeze_418
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf234 = aten.convolution_backward(buf233, relu_25, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False])
    del buf233
    del primals_136
    buf235 = buf234[0]
    buf236 = buf234[1]
    del buf234
    buf237 = empty((368, ), device='cpu', dtype=torch.float32)
    buf238 = empty((368, ), device='cpu', dtype=torch.float32)
    buf239 = empty((368, ), device='cpu', dtype=torch.float32)
    buf240 = buf235; del buf235  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35(c_void_p(buf240.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_430.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    del buf238
    del convolution_34
    del primals_45
    del relu_25
    del squeeze_67
    del unsqueeze_430
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf241 = aten.convolution_backward(buf240, relu_24, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf240
    del primals_135
    buf242 = buf241[0]
    buf243 = buf241[1]
    del buf241
    buf244 = empty((152, ), device='cpu', dtype=torch.float32)
    buf245 = empty((152, ), device='cpu', dtype=torch.float32)
    buf246 = empty_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    buf247 = buf245; del buf245  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_36(c_void_p(buf247.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()))
    del convolution_33
    del primals_43
    del squeeze_64
    del unsqueeze_442
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf248 = aten.convolution_backward(buf246, mul_152, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf246
    del mul_152
    del primals_134
    buf249 = buf248[0]
    buf250 = buf248[1]
    del buf248
    buf251 = empty_strided((8, 152, 1, 1), (152, 1, 1216, 1216), device='cpu', dtype=torch.float32)
    buf252 = reinterpret_tensor(buf251, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf251  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_37(c_void_p(buf252.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(convolution_32.data_ptr()))
    # Source Nodes: [sigmoid_5], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf253 = aten.convolution_backward(buf252, relu_23, primals_132, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf252
    del primals_132
    buf254 = buf253[0]
    buf255 = buf253[1]
    buf256 = buf253[2]
    del buf253
    buf257 = buf254; del buf254  # reuse
    cpp_fused_convolution_backward_threshold_backward_38(c_void_p(buf257.data_ptr()), c_void_p(relu_23.data_ptr()))
    del relu_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf258 = aten.convolution_backward(buf257, mean_5, primals_130, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf257
    del mean_5
    del primals_130
    buf259 = buf258[0]
    buf260 = buf258[1]
    buf261 = buf258[2]
    del buf258
    buf262 = empty((152, ), device='cpu', dtype=torch.float32)
    buf263 = empty((152, ), device='cpu', dtype=torch.float32)
    buf264 = buf249; del buf249  # reuse
    buf265 = buf263; del buf263  # reuse
    buf266 = buf264; del buf264  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_39(c_void_p(buf266.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_454.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf262.data_ptr()))
    del convolution_30
    del convolution_32
    del primals_41
    del relu_22
    del squeeze_61
    del unsqueeze_454
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf267 = aten.convolution_backward(buf266, relu_21, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False])
    del buf266
    del primals_129
    buf268 = buf267[0]
    buf269 = buf267[1]
    del buf267
    buf270 = empty((152, ), device='cpu', dtype=torch.float32)
    buf271 = empty((152, ), device='cpu', dtype=torch.float32)
    buf272 = empty((152, ), device='cpu', dtype=torch.float32)
    buf273 = buf268; del buf268  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40(c_void_p(buf273.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_466.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    del convolution_29
    del primals_39
    del relu_21
    del squeeze_58
    del unsqueeze_466
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf274 = aten.convolution_backward(buf273, relu_20, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_128
    buf275 = buf274[0]
    buf276 = buf274[1]
    del buf274
    buf277 = buf210; del buf210  # reuse
    buf278 = buf271; del buf271  # reuse
    buf279 = empty((152, ), device='cpu', dtype=torch.float32)
    buf280 = empty((152, ), device='cpu', dtype=torch.float32)
    buf281 = buf273; del buf273  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_41(c_void_p(buf277.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_478.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del buf242
    del buf275
    del convolution_28
    del primals_37
    del relu_20
    del relu_24
    del squeeze_55
    del unsqueeze_478
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf282 = aten.convolution_backward(buf281, mul_130, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf281
    del mul_130
    del primals_127
    buf283 = buf282[0]
    buf284 = buf282[1]
    del buf282
    buf285 = reinterpret_tensor(buf259, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf259  # reuse
    buf286 = reinterpret_tensor(buf285, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf285  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_42(c_void_p(buf286.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(convolution_27.data_ptr()))
    # Source Nodes: [sigmoid_4], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf287 = aten.convolution_backward(buf286, relu_19, primals_125, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf286
    del primals_125
    buf288 = buf287[0]
    buf289 = buf287[1]
    buf290 = buf287[2]
    del buf287
    buf291 = buf288; del buf288  # reuse
    cpp_fused_convolution_backward_threshold_backward_43(c_void_p(buf291.data_ptr()), c_void_p(relu_19.data_ptr()))
    del relu_19
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf292 = aten.convolution_backward(buf291, mean_4, primals_123, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf291
    del mean_4
    del primals_123
    buf293 = buf292[0]
    buf294 = buf292[1]
    buf295 = buf292[2]
    del buf292
    buf296 = buf279; del buf279  # reuse
    buf297 = empty((152, ), device='cpu', dtype=torch.float32)
    buf298 = buf283; del buf283  # reuse
    buf299 = buf297; del buf297  # reuse
    buf300 = buf298; del buf298  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_44(c_void_p(buf300.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf296.data_ptr()))
    del convolution_25
    del convolution_27
    del primals_35
    del relu_18
    del squeeze_52
    del unsqueeze_490
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf301 = aten.convolution_backward(buf300, relu_17, primals_122, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False])
    del buf300
    del primals_122
    buf302 = buf301[0]
    buf303 = buf301[1]
    del buf301
    buf304 = empty((152, ), device='cpu', dtype=torch.float32)
    buf305 = empty((152, ), device='cpu', dtype=torch.float32)
    buf306 = empty((152, ), device='cpu', dtype=torch.float32)
    buf307 = buf302; del buf302  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45(c_void_p(buf307.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_502.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del convolution_24
    del primals_33
    del relu_17
    del squeeze_49
    del unsqueeze_502
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf308 = aten.convolution_backward(buf307, relu_16, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_121
    buf309 = buf308[0]
    buf310 = buf308[1]
    del buf308
    buf311 = buf305; del buf305  # reuse
    buf312 = empty((152, ), device='cpu', dtype=torch.float32)
    buf313 = buf307; del buf307  # reuse
    buf314 = buf312; del buf312  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_46(c_void_p(buf314.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_514.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()))
    del convolution_23
    del primals_31
    del squeeze_46
    del unsqueeze_514
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf315 = aten.convolution_backward(buf313, mul_108, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf313
    del mul_108
    del primals_120
    buf316 = buf315[0]
    buf317 = buf315[1]
    del buf315
    buf318 = reinterpret_tensor(buf293, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf293  # reuse
    buf319 = reinterpret_tensor(buf318, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf318  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_47(c_void_p(buf319.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_22.data_ptr()))
    # Source Nodes: [sigmoid_3], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf320 = aten.convolution_backward(buf319, relu_15, primals_118, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf319
    del primals_118
    buf321 = buf320[0]
    buf322 = buf320[1]
    buf323 = buf320[2]
    del buf320
    buf324 = buf321; del buf321  # reuse
    cpp_fused_convolution_backward_threshold_backward_48(c_void_p(buf324.data_ptr()), c_void_p(relu_15.data_ptr()))
    del relu_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf325 = aten.convolution_backward(buf324, mean_3, primals_116, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf324
    del mean_3
    del primals_116
    buf326 = buf325[0]
    buf327 = buf325[1]
    buf328 = buf325[2]
    del buf325
    buf329 = empty((152, ), device='cpu', dtype=torch.float32)
    buf330 = empty((152, ), device='cpu', dtype=torch.float32)
    buf331 = buf316; del buf316  # reuse
    buf332 = buf330; del buf330  # reuse
    buf333 = buf331; del buf331  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_49(c_void_p(buf333.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_526.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf329.data_ptr()))
    del convolution_20
    del convolution_22
    del primals_29
    del relu_14
    del squeeze_43
    del unsqueeze_526
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf334 = aten.convolution_backward(buf333, relu_13, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False])
    del primals_115
    buf335 = buf334[0]
    buf336 = buf334[1]
    del buf334
    buf337 = empty((152, ), device='cpu', dtype=torch.float32)
    buf338 = empty((152, ), device='cpu', dtype=torch.float32)
    buf339 = empty((152, ), device='cpu', dtype=torch.float32)
    buf340 = buf335; del buf335  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50(c_void_p(buf340.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_538.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    del convolution_19
    del primals_27
    del relu_13
    del squeeze_40
    del unsqueeze_538
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf341 = aten.convolution_backward(buf340, relu_12, primals_114, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_114
    buf342 = buf341[0]
    buf343 = buf341[1]
    del buf341
    buf344 = buf277; del buf277  # reuse
    buf345 = buf338; del buf338  # reuse
    buf346 = empty((152, ), device='cpu', dtype=torch.float32)
    buf352 = empty((152, ), device='cpu', dtype=torch.float32)
    buf347 = empty((152, ), device='cpu', dtype=torch.float32)
    buf348 = buf340; del buf340  # reuse
    buf354 = buf333; del buf333  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51(c_void_p(buf344.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_550.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_562.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf354.data_ptr()))
    del buf309
    del buf342
    del buf344
    del convolution_17
    del convolution_18
    del primals_23
    del primals_25
    del relu_12
    del relu_16
    del squeeze_37
    del unsqueeze_550
    del unsqueeze_562
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf349 = aten.convolution_backward(buf348, relu_8, primals_113, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf348
    del primals_113
    buf350 = buf349[0]
    buf351 = buf349[1]
    del buf349
    buf353 = buf352; del buf352  # reuse
    cpp_fused_native_batch_norm_backward_52(c_void_p(buf353.data_ptr()), c_void_p(squeeze_34.data_ptr()))
    del squeeze_34
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf355 = aten.convolution_backward(buf354, mul_79, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf354
    del mul_79
    del primals_112
    buf356 = buf355[0]
    buf357 = buf355[1]
    del buf355
    buf358 = reinterpret_tensor(buf326, (8, 152, 1, 1), (152, 1, 1216, 1216), 0); del buf326  # reuse
    buf359 = reinterpret_tensor(buf358, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf358  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_53(c_void_p(buf359.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_16.data_ptr()))
    # Source Nodes: [sigmoid_2], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf360 = aten.convolution_backward(buf359, relu_11, primals_110, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf359
    del primals_110
    buf361 = buf360[0]
    buf362 = buf360[1]
    buf363 = buf360[2]
    del buf360
    buf364 = buf361; del buf361  # reuse
    cpp_fused_convolution_backward_threshold_backward_54(c_void_p(buf364.data_ptr()), c_void_p(relu_11.data_ptr()))
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf365 = aten.convolution_backward(buf364, mean_2, primals_108, [14], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf364
    del mean_2
    del primals_108
    buf366 = buf365[0]
    buf367 = buf365[1]
    buf368 = buf365[2]
    del buf365
    buf369 = buf346; del buf346  # reuse
    buf370 = empty((152, ), device='cpu', dtype=torch.float32)
    buf371 = buf356; del buf356  # reuse
    buf372 = buf370; del buf370  # reuse
    buf373 = buf371; del buf371  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_55(c_void_p(buf373.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_574.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf369.data_ptr()))
    del buf366
    del convolution_14
    del convolution_16
    del primals_21
    del relu_10
    del squeeze_31
    del unsqueeze_574
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf374 = aten.convolution_backward(buf373, relu_9, primals_107, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False])
    del buf373
    del primals_107
    buf375 = buf374[0]
    buf376 = buf374[1]
    del buf374
    buf377 = empty((152, ), device='cpu', dtype=torch.float32)
    buf378 = empty((152, ), device='cpu', dtype=torch.float32)
    buf379 = empty((152, ), device='cpu', dtype=torch.float32)
    buf380 = buf375; del buf375  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56(c_void_p(buf380.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_586.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    del buf378
    del convolution_13
    del primals_19
    del relu_9
    del squeeze_28
    del unsqueeze_586
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf381 = aten.convolution_backward(buf380, relu_8, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf380
    del primals_106
    buf382 = buf381[0]
    buf383 = buf381[1]
    del buf381
    buf384 = empty((56, ), device='cpu', dtype=torch.float32)
    buf385 = empty((56, ), device='cpu', dtype=torch.float32)
    buf391 = empty((56, ), device='cpu', dtype=torch.float32)
    buf386 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    buf392 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    buf387 = buf385; del buf385  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_57(c_void_p(buf387.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_598.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_610.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf392.data_ptr()))
    del buf350
    del buf382
    del convolution_11
    del convolution_12
    del primals_15
    del primals_17
    del relu_8
    del squeeze_25
    del unsqueeze_598
    del unsqueeze_610
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf388 = aten.convolution_backward(buf386, relu_4, primals_105, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf386
    del primals_105
    buf389 = buf388[0]
    buf390 = buf388[1]
    del buf388
    buf393 = buf391; del buf391  # reuse
    cpp_fused_native_batch_norm_backward_58(c_void_p(buf393.data_ptr()), c_void_p(squeeze_22.data_ptr()))
    del squeeze_22
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf394 = aten.convolution_backward(buf392, mul_50, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf392
    del mul_50
    del primals_104
    buf395 = buf394[0]
    buf396 = buf394[1]
    del buf394
    buf397 = empty_strided((8, 56, 1, 1), (56, 1, 448, 448), device='cpu', dtype=torch.float32)
    buf398 = reinterpret_tensor(buf397, (8, 56, 1, 1), (56, 1, 1, 1), 0); del buf397  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_59(c_void_p(buf398.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_10.data_ptr()))
    # Source Nodes: [sigmoid_1], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf399 = aten.convolution_backward(buf398, relu_7, primals_102, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf398
    del primals_102
    buf400 = buf399[0]
    buf401 = buf399[1]
    buf402 = buf399[2]
    del buf399
    buf403 = buf400; del buf400  # reuse
    cpp_fused_convolution_backward_threshold_backward_60(c_void_p(buf403.data_ptr()), c_void_p(relu_7.data_ptr()))
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf404 = aten.convolution_backward(buf403, mean_1, primals_100, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf403
    del mean_1
    del primals_100
    buf405 = buf404[0]
    buf406 = buf404[1]
    buf407 = buf404[2]
    del buf404
    buf408 = empty((56, ), device='cpu', dtype=torch.float32)
    buf409 = empty((56, ), device='cpu', dtype=torch.float32)
    buf410 = buf395; del buf395  # reuse
    buf411 = buf409; del buf409  # reuse
    buf412 = buf410; del buf410  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_61(c_void_p(buf412.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_622.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf408.data_ptr()))
    del buf405
    del convolution_10
    del convolution_8
    del primals_13
    del relu_6
    del squeeze_19
    del unsqueeze_622
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf413 = aten.convolution_backward(buf412, relu_5, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 7, [True, True, False])
    del buf412
    del primals_99
    buf414 = buf413[0]
    buf415 = buf413[1]
    del buf413
    buf416 = empty((56, ), device='cpu', dtype=torch.float32)
    buf417 = empty((56, ), device='cpu', dtype=torch.float32)
    buf418 = empty((56, ), device='cpu', dtype=torch.float32)
    buf419 = buf414; del buf414  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62(c_void_p(buf419.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_634.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()))
    del buf417
    del convolution_7
    del primals_11
    del relu_5
    del squeeze_16
    del unsqueeze_634
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf420 = aten.convolution_backward(buf419, relu_4, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf419
    del primals_98
    buf421 = buf420[0]
    buf422 = buf420[1]
    del buf420
    buf423 = empty((24, ), device='cpu', dtype=torch.float32)
    buf424 = empty((24, ), device='cpu', dtype=torch.float32)
    buf430 = empty((24, ), device='cpu', dtype=torch.float32)
    buf425 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    buf431 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    buf426 = buf424; del buf424  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_63(c_void_p(buf426.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_646.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_658.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf431.data_ptr()))
    del buf389
    del buf421
    del convolution_5
    del convolution_6
    del primals_7
    del primals_9
    del relu_4
    del squeeze_13
    del unsqueeze_646
    del unsqueeze_658
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf427 = aten.convolution_backward(buf425, relu, primals_97, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf425
    del primals_97
    buf428 = buf427[0]
    buf429 = buf427[1]
    del buf427
    buf432 = buf430; del buf430  # reuse
    cpp_fused_native_batch_norm_backward_64(c_void_p(buf432.data_ptr()), c_void_p(squeeze_10.data_ptr()))
    del squeeze_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf433 = aten.convolution_backward(buf431, mul_21, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf431
    del mul_21
    del primals_96
    buf434 = buf433[0]
    buf435 = buf433[1]
    del buf433
    buf436 = empty_strided((8, 24, 1, 1), (24, 1, 192, 192), device='cpu', dtype=torch.float32)
    buf437 = reinterpret_tensor(buf436, (8, 24, 1, 1), (24, 1, 1, 1), 0); del buf436  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_65(c_void_p(buf437.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_4.data_ptr()))
    # Source Nodes: [sigmoid], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf438 = aten.convolution_backward(buf437, relu_3, primals_94, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf437
    del primals_94
    buf439 = buf438[0]
    buf440 = buf438[1]
    buf441 = buf438[2]
    del buf438
    buf442 = buf439; del buf439  # reuse
    cpp_fused_convolution_backward_threshold_backward_66(c_void_p(buf442.data_ptr()), c_void_p(relu_3.data_ptr()))
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf443 = aten.convolution_backward(buf442, mean, primals_92, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf442
    del mean
    del primals_92
    buf444 = buf443[0]
    buf445 = buf443[1]
    buf446 = buf443[2]
    del buf443
    buf447 = empty((24, ), device='cpu', dtype=torch.float32)
    buf448 = empty((24, ), device='cpu', dtype=torch.float32)
    buf449 = buf434; del buf434  # reuse
    buf450 = buf448; del buf448  # reuse
    buf451 = buf449; del buf449  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_67(c_void_p(buf451.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_670.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf447.data_ptr()))
    del buf444
    del convolution_2
    del convolution_4
    del primals_5
    del relu_2
    del squeeze_7
    del unsqueeze_670
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf452 = aten.convolution_backward(buf451, relu_1, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 3, [True, True, False])
    del buf451
    del primals_91
    buf453 = buf452[0]
    buf454 = buf452[1]
    del buf452
    buf455 = empty((24, ), device='cpu', dtype=torch.float32)
    buf456 = empty((24, ), device='cpu', dtype=torch.float32)
    buf457 = empty((24, ), device='cpu', dtype=torch.float32)
    buf458 = buf453; del buf453  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68(c_void_p(buf458.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_682.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()))
    del buf456
    del convolution_1
    del primals_3
    del relu_1
    del squeeze_4
    del unsqueeze_682
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf459 = aten.convolution_backward(buf458, relu, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf458
    del primals_90
    buf460 = buf459[0]
    buf461 = buf459[1]
    del buf459
    buf462 = empty((32, ), device='cpu', dtype=torch.float32)
    buf463 = empty((32, ), device='cpu', dtype=torch.float32)
    buf464 = buf428; del buf428  # reuse
    buf465 = buf463; del buf463  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_69(c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_694.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf462.data_ptr()))
    del buf460
    del convolution
    del primals_1
    del relu
    del squeeze_1
    del unsqueeze_694
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf466 = aten.convolution_backward(buf464, primals_319, primals_89, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf464
    del primals_319
    del primals_89
    buf467 = buf466[1]
    return (buf465, buf462, buf457, buf455, buf450, buf447, buf432, buf423, buf426, buf423, buf418, buf416, buf411, buf408, buf393, buf384, buf387, buf384, buf379, buf377, buf372, buf369, buf353, buf345, buf347, buf345, buf339, buf337, buf332, buf329, buf314, buf311, buf306, buf304, buf299, buf296, buf280, buf278, buf272, buf270, buf265, buf262, buf247, buf244, buf239, buf237, buf232, buf229, buf213, buf205, buf207, buf205, buf199, buf197, buf192, buf189, buf174, buf171, buf166, buf164, buf159, buf156, buf140, buf138, buf132, buf130, buf125, buf122, buf107, buf104, buf99, buf97, buf92, buf89, buf73, buf71, buf65, buf63, buf58, buf55, buf39, buf36, buf31, buf29, buf24, buf21, buf5, buf3, buf467, buf461, buf454, buf445, buf446, buf440, buf441, buf435, buf429, buf422, buf415, buf406, buf407, buf401, buf402, buf396, buf390, buf383, buf376, buf367, buf368, buf362, buf363, buf357, buf351, buf343, buf336, buf327, buf328, buf322, buf323, buf317, buf310, buf303, buf294, buf295, buf289, buf290, buf284, buf276, buf269, buf260, buf261, buf255, buf256, buf250, buf243, buf236, buf227, buf228, buf222, buf223, buf217, buf211, buf203, buf196, buf187, buf188, buf182, buf183, buf177, buf170, buf163, buf154, buf155, buf149, buf150, buf144, buf136, buf129, buf120, buf121, buf115, buf116, buf110, buf103, buf96, buf87, buf88, buf82, buf83, buf77, buf69, buf62, buf53, buf54, buf48, buf49, buf43, buf35, buf28, buf19, buf20, buf14, buf15, buf9, reinterpret_tensor(buf1, (1000, 368), (368, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((24, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((24, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((56, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((6, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((56, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((152, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((14, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((152, 14, 1, 1), (14, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((152, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((152, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((152, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((368, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((38, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((368, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((368, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((368, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((368, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((368, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((368, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((368, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    mul_21 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 56, 56, 56), (175616, 1, 3136, 56), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 56, 56, 56), (175616, 1, 3136, 56), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    mul_50 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 152, 28, 28), (119168, 1, 4256, 152), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 152, 28, 28), (119168, 1, 4256, 152), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 152, 1, 1), (152, 1, 152, 152), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 14, 1, 1), (14, 1, 14, 14), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 152, 1, 1), (152, 1, 152, 152), device='cpu', dtype=torch.float32)
    mul_79 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 152, 1, 1), (152, 1, 152, 152), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 38, 1, 1), (38, 1, 38, 38), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 152, 1, 1), (152, 1, 152, 152), device='cpu', dtype=torch.float32)
    mul_108 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 152, 1, 1), (152, 1, 152, 152), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((8, 38, 1, 1), (38, 1, 38, 38), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 152, 1, 1), (152, 1, 152, 152), device='cpu', dtype=torch.float32)
    mul_130 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 152, 1, 1), (152, 1, 152, 152), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((8, 38, 1, 1), (38, 1, 38, 38), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 152, 1, 1), (152, 1, 152, 152), device='cpu', dtype=torch.float32)
    mul_152 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((8, 152, 14, 14), (29792, 1, 2128, 152), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 368, 14, 14), (72128, 1, 5152, 368), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((8, 368, 14, 14), (72128, 1, 5152, 368), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((8, 38, 1, 1), (38, 1, 38, 38), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    mul_174 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((8, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    mul_203 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((8, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    mul_225 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_36 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_38 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    relu_39 = rand_strided((8, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    mul_247 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_40 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_41 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_42 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    relu_43 = rand_strided((8, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    mul_269 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_44 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_46 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    relu_47 = rand_strided((8, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    mul_291 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_48 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_49 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_50 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    mean_12 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    relu_51 = rand_strided((8, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 368, 1, 1), (368, 1, 368, 368), device='cpu', dtype=torch.float32)
    mul_313 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((368, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 368), (368, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 368), (368, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 368, 7, 7), (18032, 1, 2576, 368), device='cpu', dtype=torch.bool)
    unsqueeze_178 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_214 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 368, 1, 1), (368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 152, 1, 1), (152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_90, primals_91, primals_92, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_149, primals_150, primals_151, primals_152, primals_154, primals_156, primals_157, primals_158, primals_159, primals_161, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_319, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, mean, relu_3, convolution_4, mul_21, convolution_5, squeeze_10, convolution_6, squeeze_13, relu_4, convolution_7, squeeze_16, relu_5, convolution_8, squeeze_19, relu_6, mean_1, relu_7, convolution_10, mul_50, convolution_11, squeeze_22, convolution_12, squeeze_25, relu_8, convolution_13, squeeze_28, relu_9, convolution_14, squeeze_31, relu_10, mean_2, relu_11, convolution_16, mul_79, convolution_17, squeeze_34, convolution_18, squeeze_37, relu_12, convolution_19, squeeze_40, relu_13, convolution_20, squeeze_43, relu_14, mean_3, relu_15, convolution_22, mul_108, convolution_23, squeeze_46, relu_16, convolution_24, squeeze_49, relu_17, convolution_25, squeeze_52, relu_18, mean_4, relu_19, convolution_27, mul_130, convolution_28, squeeze_55, relu_20, convolution_29, squeeze_58, relu_21, convolution_30, squeeze_61, relu_22, mean_5, relu_23, convolution_32, mul_152, convolution_33, squeeze_64, relu_24, convolution_34, squeeze_67, relu_25, convolution_35, squeeze_70, relu_26, mean_6, relu_27, convolution_37, mul_174, convolution_38, squeeze_73, convolution_39, squeeze_76, relu_28, convolution_40, squeeze_79, relu_29, convolution_41, squeeze_82, relu_30, mean_7, relu_31, convolution_43, mul_203, convolution_44, squeeze_85, relu_32, convolution_45, squeeze_88, relu_33, convolution_46, squeeze_91, relu_34, mean_8, relu_35, convolution_48, mul_225, convolution_49, squeeze_94, relu_36, convolution_50, squeeze_97, relu_37, convolution_51, squeeze_100, relu_38, mean_9, relu_39, convolution_53, mul_247, convolution_54, squeeze_103, relu_40, convolution_55, squeeze_106, relu_41, convolution_56, squeeze_109, relu_42, mean_10, relu_43, convolution_58, mul_269, convolution_59, squeeze_112, relu_44, convolution_60, squeeze_115, relu_45, convolution_61, squeeze_118, relu_46, mean_11, relu_47, convolution_63, mul_291, convolution_64, squeeze_121, relu_48, convolution_65, squeeze_124, relu_49, convolution_66, squeeze_127, relu_50, mean_12, relu_51, convolution_68, mul_313, convolution_69, squeeze_130, clone, permute_1, le, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('regnety_002', benchmark_compiled_module)
