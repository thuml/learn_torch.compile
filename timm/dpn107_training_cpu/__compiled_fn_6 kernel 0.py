
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


cpp_fused_add_convolution_backward_div_native_batch_norm_backward_slice_backward_threshold_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (2688L*x2) + (131712L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2688L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2688L*x2) + (131712L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
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
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2688L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2688L*x1) + (131712L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (2688L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2688L*x1) + (131712L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        tmp25.store(out_ptr2 + static_cast<long>(x2 + (2688L*x1) + (131712L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2176L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(2048);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>(512L + x1 + (2688L*x2) + (131712L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = out_ptr2[static_cast<long>(x1 + (2688L*x2) + (131712L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        out_ptr3[static_cast<long>(x2 + (49L*x1) + (106624L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1600L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1600L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1600L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1600L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2560L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2560L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2560L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2560L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
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
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(2048);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(384L + x2 + (2688L*x1) + (131712L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(384L + x2 + (2560L*x1) + (125440L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr6[static_cast<long>(x2 + (2688L*x1) + (131712L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(x2 + (2560L*x1) + (125440L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        out_ptr2[static_cast<long>(x1 + (49L*x2) + (106624L*x0))] = tmp17;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1600L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1600L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1600L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1600L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_6 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2432L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2432L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2432L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2432L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2432L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2432L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2432L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2432L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (2432L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2432L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
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
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(2048);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(256L + x2 + (2688L*x1) + (131712L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(256L + x2 + (2560L*x1) + (125440L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_out_ptr0[static_cast<long>(256L + x2 + (2432L*x1) + (119168L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr6[static_cast<long>(x2 + (2688L*x1) + (131712L*x0))];
                            auto tmp15 = in_ptr7[static_cast<long>(x2 + (2560L*x1) + (125440L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_out_ptr0[static_cast<long>(x2 + (2432L*x1) + (119168L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr2[static_cast<long>(x1 + (49L*x2) + (106624L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1600L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1600L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1600L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1600L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_slice_backward_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(2048);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x2 + (2688L*x1) + (131712L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x2 + (2560L*x1) + (125440L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr2[static_cast<long>(x2 + (2432L*x1) + (119168L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr0[static_cast<long>(x2 + (2688L*x1) + (131712L*x0))];
                            auto tmp15 = in_ptr1[static_cast<long>(x2 + (2560L*x1) + (125440L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_ptr2[static_cast<long>(x2 + (2432L*x1) + (119168L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr0[static_cast<long>(x1 + (49L*x2) + (112896L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2432L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2432L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2432L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2432L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2432L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (2432L*x1)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp11 = to_float_mask(tmp10 <= tmp2);
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp11);
                        auto tmp14 = tmp13 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp14;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2432L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp3 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                    tmp4.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_11 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2432L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2432L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2432L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2432L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (2432L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (2432L*x0)));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    auto tmp25 = to_float_mask(tmp24 <= tmp2);
                    auto tmp27 = decltype(tmp2)::blendv(tmp26, tmp2, tmp25);
                    auto tmp29 = tmp28 * tmp11;
                    auto tmp30 = tmp29 * tmp14;
                    auto tmp31 = tmp8 * tmp30;
                    auto tmp32 = tmp27 - tmp31;
                    auto tmp34 = tmp33 * tmp11;
                    auto tmp35 = tmp32 - tmp34;
                    auto tmp37 = tmp13 * tmp36;
                    auto tmp38 = tmp35 * tmp37;
                    auto tmp39 = tmp23 + tmp38;
                    tmp39.store(in_out_ptr0 + static_cast<long>(x1 + (2432L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1088L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_out_ptr0[static_cast<long>(1344L + x1 + (2432L*x2) + (476672L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_out_ptr0[static_cast<long>(x1 + (2432L*x2) + (476672L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (213248L*x0))] = tmp13;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2368L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2368L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2368L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2368L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2368L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2368L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2368L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2368L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (2368L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2368L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(1280L + x2 + (2432L*x1) + (476672L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(1280L + x2 + (2368L*x1) + (464128L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr6[static_cast<long>(x2 + (2432L*x1) + (476672L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(x2 + (2368L*x1) + (464128L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp17;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_17 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2304L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2304L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2304L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2304L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2304L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (2304L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(1216L + x2 + (2432L*x1) + (476672L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(1216L + x2 + (2368L*x1) + (464128L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_out_ptr0[static_cast<long>(1216L + x2 + (2304L*x1) + (451584L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr6[static_cast<long>(x2 + (2432L*x1) + (476672L*x0))];
                            auto tmp15 = in_ptr7[static_cast<long>(x2 + (2368L*x1) + (464128L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_out_ptr0[static_cast<long>(x2 + (2304L*x1) + (451584L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp21;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_20 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2240L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(1152L + x2 + (2432L*x1) + (476672L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(1152L + x2 + (2368L*x1) + (464128L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr8[static_cast<long>(1152L + x2 + (2304L*x1) + (451584L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_out_ptr0[static_cast<long>(1152L + x2 + (2240L*x1) + (439040L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr6[static_cast<long>(x2 + (2432L*x1) + (476672L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x2 + (2368L*x1) + (464128L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr8[static_cast<long>(x2 + (2304L*x1) + (451584L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x2 + (2240L*x1) + (439040L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp25;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_23 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2176L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2176L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2176L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2176L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2176L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2176L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2176L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2176L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (2176L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2176L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (2432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (2368L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (2304L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (2240L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2176L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(1024L + x1 + (2432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(1024L + x1 + (2368L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(1024L + x1 + (2304L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(1024L + x1 + (2240L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(1024L + x1 + (2176L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr3[static_cast<long>(64L + x2 + (1152L*x1) + (225792L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = out_ptr2[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        out_ptr4[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp13;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_26 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2112L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2112L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2112L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2112L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2112L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2112L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (2112L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2112L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(x2 + (1152L*x1) + (225792L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(1024L + x2 + (2112L*x1) + (413952L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr7[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(x2 + (2112L*x1) + (413952L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp17;
                    }
                }
            }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_29 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-64L) + x2 + (1152L*x1) + (225792L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(960L + x2 + (2112L*x1) + (413952L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_out_ptr0[static_cast<long>(960L + x2 + (2048L*x1) + (401408L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr8[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp15 = in_ptr7[static_cast<long>(x2 + (2112L*x1) + (413952L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_out_ptr0[static_cast<long>(x2 + (2048L*x1) + (401408L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp21;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_32 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1984L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1984L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1984L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1984L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1984L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1984L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1984L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1984L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1984L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1984L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-128L) + x2 + (1152L*x1) + (225792L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(896L + x2 + (2112L*x1) + (413952L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr8[static_cast<long>(896L + x2 + (2048L*x1) + (401408L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_out_ptr0[static_cast<long>(896L + x2 + (1984L*x1) + (388864L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr9[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x2 + (2112L*x1) + (413952L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr8[static_cast<long>(x2 + (2048L*x1) + (401408L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x2 + (1984L*x1) + (388864L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp25;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
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
                       float* out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (2112L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (1984L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(1024L + x1 + (2112L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(1024L + x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(1024L + x1 + (1984L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(1024L + x1 + (1920L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(out_ptr2 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>((-192L) + x2 + (896L*x1) + (175616L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_out_ptr2[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        out_ptr3[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_38 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1856L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1856L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1856L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1856L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1856L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1856L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1856L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1856L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1856L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1856L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-256L) + x2 + (896L*x1) + (175616L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(768L + x2 + (1856L*x1) + (363776L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr7[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(x2 + (1856L*x1) + (363776L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp17;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_41 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1792L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1792L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1792L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1792L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1792L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1792L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1792L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1792L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-320L) + x2 + (896L*x1) + (175616L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(704L + x2 + (1856L*x1) + (363776L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_out_ptr0[static_cast<long>(704L + x2 + (1792L*x1) + (351232L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr8[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp15 = in_ptr7[static_cast<long>(x2 + (1856L*x1) + (363776L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_out_ptr0[static_cast<long>(x2 + (1792L*x1) + (351232L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp21;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_44 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1728L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1728L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1728L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1728L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1728L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1728L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1728L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1728L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1728L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1728L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-384L) + x2 + (896L*x1) + (175616L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(640L + x2 + (1856L*x1) + (363776L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr8[static_cast<long>(640L + x2 + (1792L*x1) + (351232L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_out_ptr0[static_cast<long>(640L + x2 + (1728L*x1) + (338688L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr9[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x2 + (1856L*x1) + (363776L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr8[static_cast<long>(x2 + (1792L*x1) + (351232L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x2 + (1728L*x1) + (338688L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp25;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
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
                       float* out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1664L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1664L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1664L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1664L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1664L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1664L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1664L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1664L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1664L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1664L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1856L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1792L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (1728L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1664L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(1024L + x1 + (1856L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(1024L + x1 + (1792L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(1024L + x1 + (1728L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(1024L + x1 + (1664L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(out_ptr2 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>((-448L) + x2 + (640L*x1) + (125440L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_out_ptr2[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        out_ptr3[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_50 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1600L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1600L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1600L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1600L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1600L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1600L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-512L) + x2 + (640L*x1) + (125440L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(512L + x2 + (1600L*x1) + (313600L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr7[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(x2 + (1600L*x1) + (313600L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_53 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1536L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1536L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-576L) + x2 + (640L*x1) + (125440L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(448L + x2 + (1600L*x1) + (313600L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_out_ptr0[static_cast<long>(448L + x2 + (1536L*x1) + (301056L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr8[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp15 = in_ptr7[static_cast<long>(x2 + (1600L*x1) + (313600L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_out_ptr0[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_56 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1472L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1472L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1472L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1472L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1472L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1472L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1472L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1472L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1472L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1472L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-640L) + x2 + (640L*x1) + (125440L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(384L + x2 + (1600L*x1) + (313600L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr8[static_cast<long>(384L + x2 + (1536L*x1) + (301056L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_out_ptr0[static_cast<long>(384L + x2 + (1472L*x1) + (288512L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr9[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x2 + (1600L*x1) + (313600L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr8[static_cast<long>(x2 + (1536L*x1) + (301056L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x2 + (1472L*x1) + (288512L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp25;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
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
                       float* out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1408L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1408L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1408L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1408L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1408L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1408L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1408L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1408L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1408L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1600L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (1472L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1408L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(1024L + x1 + (1600L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(1024L + x1 + (1536L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(1024L + x1 + (1472L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(1024L + x1 + (1408L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr2[static_cast<long>((-704L) + x2 + (384L*x1) + (75264L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_out_ptr2[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        out_ptr3[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_62 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1344L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1344L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1344L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1344L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1344L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1344L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1344L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1344L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-768L) + x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(256L + x2 + (1344L*x1) + (263424L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr7[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(x2 + (1344L*x1) + (263424L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_65 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1280L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1280L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1280L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1280L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-832L) + x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(192L + x2 + (1344L*x1) + (263424L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_out_ptr0[static_cast<long>(192L + x2 + (1280L*x1) + (250880L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr8[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp15 = in_ptr7[static_cast<long>(x2 + (1344L*x1) + (263424L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_out_ptr0[static_cast<long>(x2 + (1280L*x1) + (250880L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_68 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1216L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1216L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1216L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1216L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1216L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1216L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1216L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1216L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1088L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-896L) + x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(128L + x2 + (1344L*x1) + (263424L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr8[static_cast<long>(128L + x2 + (1280L*x1) + (250880L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_out_ptr0[static_cast<long>(128L + x2 + (1216L*x1) + (238336L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr9[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x2 + (1344L*x1) + (263424L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr8[static_cast<long>(x2 + (1280L*x1) + (250880L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x2 + (1216L*x1) + (238336L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (213248L*x0))] = tmp25;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (800L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (800L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(800L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(800L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (800L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (800L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_slice_backward_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(1024);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>((-1024L) + x2 + (384L*x1) + (75264L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x2 + (1344L*x1) + (263424L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr2[static_cast<long>(x2 + (1280L*x1) + (250880L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr3[static_cast<long>(x2 + (1216L*x1) + (238336L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr4[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                            auto tmp17 = in_ptr1[static_cast<long>(x2 + (1344L*x1) + (263424L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr2[static_cast<long>(x2 + (1280L*x1) + (250880L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_ptr3[static_cast<long>(x2 + (1216L*x1) + (238336L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (225792L*x0))] = tmp25;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp11 = to_float_mask(tmp10 <= tmp2);
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp11);
                        auto tmp14 = tmp13 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp14;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp3 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                    tmp4.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_73 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    auto tmp25 = to_float_mask(tmp24 <= tmp2);
                    auto tmp27 = decltype(tmp2)::blendv(tmp26, tmp2, tmp25);
                    auto tmp29 = tmp28 * tmp11;
                    auto tmp30 = tmp29 * tmp14;
                    auto tmp31 = tmp8 * tmp30;
                    auto tmp32 = tmp27 - tmp31;
                    auto tmp34 = tmp33 * tmp11;
                    auto tmp35 = tmp32 - tmp34;
                    auto tmp37 = tmp13 * tmp36;
                    auto tmp38 = tmp35 * tmp37;
                    auto tmp39 = tmp23 + tmp38;
                    tmp39.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(512);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_out_ptr0[static_cast<long>(576L + x1 + (1152L*x2) + (903168L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_out_ptr0[static_cast<long>(x1 + (1152L*x2) + (903168L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        out_ptr0[static_cast<long>(x2 + (784L*x1) + (451584L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1088L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1088L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1088L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1088L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1088L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1088L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1088L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1088L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1088L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(512);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(512L + x2 + (1152L*x1) + (903168L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(512L + x2 + (1088L*x1) + (852992L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr6[static_cast<long>(x2 + (1152L*x1) + (903168L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(x2 + (1088L*x1) + (852992L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        out_ptr2[static_cast<long>(x1 + (784L*x2) + (451584L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_79 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(512);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(448L + x2 + (1152L*x1) + (903168L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(448L + x2 + (1088L*x1) + (852992L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_out_ptr0[static_cast<long>(448L + x2 + (1024L*x1) + (802816L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr6[static_cast<long>(x2 + (1152L*x1) + (903168L*x0))];
                            auto tmp15 = in_ptr7[static_cast<long>(x2 + (1088L*x1) + (852992L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (802816L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr2[static_cast<long>(x1 + (784L*x2) + (451584L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_82 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(512);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(384L + x2 + (1152L*x1) + (903168L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(384L + x2 + (1088L*x1) + (852992L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr8[static_cast<long>(384L + x2 + (1024L*x1) + (802816L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_out_ptr0[static_cast<long>(384L + x2 + (960L*x1) + (752640L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr6[static_cast<long>(x2 + (1152L*x1) + (903168L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x2 + (1088L*x1) + (852992L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr8[static_cast<long>(x2 + (1024L*x1) + (802816L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x2 + (960L*x1) + (752640L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr2[static_cast<long>(x1 + (784L*x2) + (451584L*x0))] = tmp25;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_85 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (896L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1088L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(512L + x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(512L + x1 + (1088L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(512L + x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(512L + x1 + (960L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(512L + x1 + (896L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(512);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = out_ptr3[static_cast<long>((-192L) + x2 + (384L*x1) + (301056L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = out_ptr2[static_cast<long>(x2 + (512L*x1) + (401408L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        out_ptr4[static_cast<long>(x1 + (784L*x2) + (451584L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_88 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(832L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (832L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (832L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (832L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(832L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (832L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (832L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (832L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(832L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(512);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-256L) + x2 + (384L*x1) + (301056L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(256L + x2 + (832L*x1) + (652288L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr7[static_cast<long>(x2 + (512L*x1) + (401408L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(x2 + (832L*x1) + (652288L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        out_ptr2[static_cast<long>(x1 + (784L*x2) + (451584L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_91 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(512);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-320L) + x2 + (384L*x1) + (301056L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(192L + x2 + (832L*x1) + (652288L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_out_ptr0[static_cast<long>(192L + x2 + (768L*x1) + (602112L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr8[static_cast<long>(x2 + (512L*x1) + (401408L*x0))];
                            auto tmp15 = in_ptr7[static_cast<long>(x2 + (832L*x1) + (652288L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (602112L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr2[static_cast<long>(x1 + (784L*x2) + (451584L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_94 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(704L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (704L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (704L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (704L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(704L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (704L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (704L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (704L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (704L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(704L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(512);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>((-384L) + x2 + (384L*x1) + (301056L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(128L + x2 + (832L*x1) + (652288L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr8[static_cast<long>(128L + x2 + (768L*x1) + (602112L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_out_ptr0[static_cast<long>(128L + x2 + (704L*x1) + (551936L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr9[static_cast<long>(x2 + (512L*x1) + (401408L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x2 + (832L*x1) + (652288L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr8[static_cast<long>(x2 + (768L*x1) + (602112L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x2 + (704L*x1) + (551936L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr2[static_cast<long>(x1 + (784L*x2) + (451584L*x0))] = tmp25;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (400L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (400L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (400L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (400L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_slice_backward_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(512);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>((-512L) + x2 + (384L*x1) + (301056L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x2 + (832L*x1) + (652288L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (602112L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr3[static_cast<long>(x2 + (704L*x1) + (551936L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr4[static_cast<long>(x2 + (512L*x1) + (401408L*x0))];
                            auto tmp17 = in_ptr1[static_cast<long>(x2 + (832L*x1) + (652288L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (602112L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_ptr3[static_cast<long>(x2 + (704L*x1) + (551936L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr0[static_cast<long>(x1 + (784L*x2) + (501760L*x0))] = tmp25;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (376L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (376L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (376L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (376L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (376L*x1)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        auto tmp11 = to_float_mask(tmp10 <= tmp2);
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp11);
                        auto tmp14 = tmp13 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        tmp_acc3_vec = tmp_acc3_vec + tmp14;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp3 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                    tmp4.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_99 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(376L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (376L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (376L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (376L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (376L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (376L*x0)));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    auto tmp25 = to_float_mask(tmp24 <= tmp2);
                    auto tmp27 = decltype(tmp2)::blendv(tmp26, tmp2, tmp25);
                    auto tmp29 = tmp28 * tmp11;
                    auto tmp30 = tmp29 * tmp14;
                    auto tmp31 = tmp8 * tmp30;
                    auto tmp32 = tmp27 - tmp31;
                    auto tmp34 = tmp33 * tmp11;
                    auto tmp35 = tmp32 - tmp34;
                    auto tmp37 = tmp13 * tmp36;
                    auto tmp38 = tmp35 * tmp37;
                    auto tmp39 = tmp23 + tmp38;
                    tmp39.store(in_out_ptr0 + static_cast<long>(x1 + (376L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(276L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_out_ptr0[static_cast<long>(100L + x1 + (376L*x2) + (1179136L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_out_ptr0[static_cast<long>(x1 + (376L*x2) + (1179136L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        out_ptr0[static_cast<long>(x2 + (3136L*x1) + (865536L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_100 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_101 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(352L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (356L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (356L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (356L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(352L); x0<static_cast<long>(356L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (356L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (356L*x1))];
                        auto tmp5 = in_ptr2[static_cast<long>(x0 + (356L*x1))];
                        auto tmp6 = in_ptr3[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp4 = tmp2 ? tmp1 : tmp3;
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                        tmp_acc0 = tmp_acc0 + tmp4;
                        tmp_acc1 = tmp_acc1 + tmp8;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (356L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (356L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (356L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (356L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(352L); x1<static_cast<long>(356L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (356L*x0))];
                    auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (356L*x0))];
                    auto tmp5 = in_ptr2[static_cast<long>(x1 + (356L*x0))];
                    auto tmp6 = in_ptr3[static_cast<long>(x1)];
                    auto tmp8 = out_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr4[static_cast<long>(x1)];
                    auto tmp16 = out_ptr0[static_cast<long>(x1)];
                    auto tmp19 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp9 = static_cast<float>(3.985969387755102e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp11);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = decltype(tmp7)(tmp7 * tmp13);
                    auto tmp15 = decltype(tmp4)(tmp4 - tmp14);
                    auto tmp17 = decltype(tmp16)(tmp16 * tmp9);
                    auto tmp18 = decltype(tmp15)(tmp15 - tmp17);
                    auto tmp20 = decltype(tmp11)(tmp11 * tmp19);
                    auto tmp21 = decltype(tmp18)(tmp18 * tmp20);
                    in_out_ptr0[static_cast<long>(x1 + (356L*x0))] = tmp21;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(352L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(352L); x0<static_cast<long>(356L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(276L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(80L + x2 + (376L*x1) + (1179136L*x0))];
                            auto tmp5 = in_out_ptr0[static_cast<long>(80L + x2 + (356L*x1) + (1116416L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr6[static_cast<long>(x2 + (376L*x1) + (1179136L*x0))];
                            auto tmp13 = in_out_ptr0[static_cast<long>(x2 + (356L*x1) + (1116416L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        out_ptr2[static_cast<long>(x1 + (3136L*x2) + (865536L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_104 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_105 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (336L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (336L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (336L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (336L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(276L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(60L + x2 + (376L*x1) + (1179136L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(60L + x2 + (356L*x1) + (1116416L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_out_ptr0[static_cast<long>(60L + x2 + (336L*x1) + (1053696L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr6[static_cast<long>(x2 + (376L*x1) + (1179136L*x0))];
                            auto tmp15 = in_ptr7[static_cast<long>(x2 + (356L*x1) + (1116416L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_out_ptr0[static_cast<long>(x2 + (336L*x1) + (1053696L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        out_ptr2[static_cast<long>(x1 + (3136L*x2) + (865536L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_106 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_107 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_108 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(312L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (316L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (316L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (316L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(312L); x0<static_cast<long>(316L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (316L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (316L*x1))];
                        auto tmp5 = in_ptr2[static_cast<long>(x0 + (316L*x1))];
                        auto tmp6 = in_ptr3[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp4 = tmp2 ? tmp1 : tmp3;
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                        tmp_acc0 = tmp_acc0 + tmp4;
                        tmp_acc1 = tmp_acc1 + tmp8;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(312L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (316L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (316L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (316L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (316L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(312L); x1<static_cast<long>(316L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (316L*x0))];
                    auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (316L*x0))];
                    auto tmp5 = in_ptr2[static_cast<long>(x1 + (316L*x0))];
                    auto tmp6 = in_ptr3[static_cast<long>(x1)];
                    auto tmp8 = out_ptr1[static_cast<long>(x1)];
                    auto tmp11 = in_ptr4[static_cast<long>(x1)];
                    auto tmp16 = out_ptr0[static_cast<long>(x1)];
                    auto tmp19 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp9 = static_cast<float>(3.985969387755102e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp11);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = decltype(tmp7)(tmp7 * tmp13);
                    auto tmp15 = decltype(tmp4)(tmp4 - tmp14);
                    auto tmp17 = decltype(tmp16)(tmp16 * tmp9);
                    auto tmp18 = decltype(tmp15)(tmp15 - tmp17);
                    auto tmp20 = decltype(tmp11)(tmp11 * tmp19);
                    auto tmp21 = decltype(tmp18)(tmp18 * tmp20);
                    in_out_ptr0[static_cast<long>(x1 + (316L*x0))] = tmp21;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(312L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(312L); x0<static_cast<long>(316L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(276L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr6[static_cast<long>(40L + x2 + (376L*x1) + (1179136L*x0))];
                            auto tmp5 = in_ptr7[static_cast<long>(40L + x2 + (356L*x1) + (1116416L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr8[static_cast<long>(40L + x2 + (336L*x1) + (1053696L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_out_ptr0[static_cast<long>(40L + x2 + (316L*x1) + (990976L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr6[static_cast<long>(x2 + (376L*x1) + (1179136L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x2 + (356L*x1) + (1116416L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr8[static_cast<long>(x2 + (336L*x1) + (1053696L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x2 + (316L*x1) + (990976L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr2[static_cast<long>(x1 + (3136L*x2) + (865536L*x0))] = tmp25;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_109 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_110 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_slice_backward_111 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(296L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(256);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x2 + (376L*x1) + (1179136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x2 + (356L*x1) + (1116416L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr2[static_cast<long>(x2 + (336L*x1) + (1053696L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr3[static_cast<long>(x2 + (316L*x1) + (990976L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr0[static_cast<long>(x2 + (376L*x1) + (1179136L*x0))];
                            auto tmp17 = in_ptr1[static_cast<long>(x2 + (356L*x1) + (1116416L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr2[static_cast<long>(x2 + (336L*x1) + (1053696L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_ptr3[static_cast<long>(x2 + (316L*x1) + (990976L*x0))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        out_ptr0[static_cast<long>(x1 + (3136L*x2) + (928256L*x0))] = tmp25;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_112 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp7 = tmp5 * tmp6;
                        auto tmp9 = to_float_mask(tmp8 <= tmp2);
                        auto tmp11 = decltype(tmp2)::blendv(tmp10, tmp2, tmp9);
                        auto tmp12 = tmp11 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        tmp_acc2_vec = tmp_acc2_vec + tmp11;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp3 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                    tmp4.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_113 = async_compile.cpp('''
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
                       const float* in_ptr10)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = static_cast<float>(3.985969387755102e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp12 = tmp11 * tmp11;
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp15 = tmp5 - tmp14;
                    auto tmp17 = tmp16 * tmp9;
                    auto tmp18 = tmp15 - tmp17;
                    auto tmp20 = tmp11 * tmp19;
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp23 = to_float_mask(tmp22 <= tmp2);
                    auto tmp25 = decltype(tmp2)::blendv(tmp24, tmp2, tmp23);
                    auto tmp27 = tmp26 * tmp9;
                    auto tmp28 = tmp27 * tmp12;
                    auto tmp29 = tmp6 * tmp28;
                    auto tmp30 = tmp25 - tmp29;
                    auto tmp32 = tmp31 * tmp9;
                    auto tmp33 = tmp30 - tmp32;
                    auto tmp35 = tmp11 * tmp34;
                    auto tmp36 = tmp33 * tmp35;
                    auto tmp37 = tmp21 + tmp36;
                    tmp37.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_114 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_668, convolution, squeeze_1, relu, getitem_3, squeeze_4, relu_1, relu_2, convolution_2, squeeze_10, relu_3, convolution_3, squeeze_13, relu_4, cat_1, squeeze_16, relu_5, convolution_5, squeeze_19, relu_6, convolution_6, squeeze_22, relu_7, cat_3, squeeze_25, relu_8, convolution_8, squeeze_28, relu_9, convolution_9, squeeze_31, relu_10, cat_5, squeeze_34, relu_11, convolution_11, squeeze_37, relu_12, convolution_12, squeeze_40, relu_13, cat_7, squeeze_43, relu_14, relu_15, convolution_15, squeeze_49, relu_16, convolution_16, squeeze_52, relu_17, cat_9, squeeze_55, relu_18, convolution_18, squeeze_58, relu_19, convolution_19, squeeze_61, relu_20, cat_11, squeeze_64, relu_21, convolution_21, squeeze_67, relu_22, convolution_22, squeeze_70, relu_23, cat_13, squeeze_73, relu_24, convolution_24, squeeze_76, relu_25, convolution_25, squeeze_79, relu_26, cat_15, squeeze_82, relu_27, convolution_27, squeeze_85, relu_28, convolution_28, squeeze_88, relu_29, cat_17, squeeze_91, relu_30, convolution_30, squeeze_94, relu_31, convolution_31, squeeze_97, relu_32, cat_19, squeeze_100, relu_33, convolution_33, squeeze_103, relu_34, convolution_34, squeeze_106, relu_35, cat_21, squeeze_109, relu_36, convolution_36, squeeze_112, relu_37, convolution_37, squeeze_115, relu_38, cat_23, squeeze_118, relu_39, relu_40, convolution_40, squeeze_124, relu_41, convolution_41, squeeze_127, relu_42, cat_25, squeeze_130, relu_43, convolution_43, squeeze_133, relu_44, convolution_44, squeeze_136, relu_45, cat_27, squeeze_139, relu_46, convolution_46, squeeze_142, relu_47, convolution_47, squeeze_145, relu_48, cat_29, squeeze_148, relu_49, convolution_49, squeeze_151, relu_50, convolution_50, squeeze_154, relu_51, cat_31, squeeze_157, relu_52, convolution_52, squeeze_160, relu_53, convolution_53, squeeze_163, relu_54, cat_33, squeeze_166, relu_55, convolution_55, squeeze_169, relu_56, convolution_56, squeeze_172, relu_57, cat_35, squeeze_175, relu_58, convolution_58, squeeze_178, relu_59, convolution_59, squeeze_181, relu_60, cat_37, squeeze_184, relu_61, convolution_61, squeeze_187, relu_62, convolution_62, squeeze_190, relu_63, cat_39, squeeze_193, relu_64, convolution_64, squeeze_196, relu_65, convolution_65, squeeze_199, relu_66, cat_41, squeeze_202, relu_67, convolution_67, squeeze_205, relu_68, convolution_68, squeeze_208, relu_69, cat_43, squeeze_211, relu_70, convolution_70, squeeze_214, relu_71, convolution_71, squeeze_217, relu_72, cat_45, squeeze_220, relu_73, convolution_73, squeeze_223, relu_74, convolution_74, squeeze_226, relu_75, cat_47, squeeze_229, relu_76, convolution_76, squeeze_232, relu_77, convolution_77, squeeze_235, relu_78, cat_49, squeeze_238, relu_79, convolution_79, squeeze_241, relu_80, convolution_80, squeeze_244, relu_81, cat_51, squeeze_247, relu_82, convolution_82, squeeze_250, relu_83, convolution_83, squeeze_253, relu_84, cat_53, squeeze_256, relu_85, convolution_85, squeeze_259, relu_86, convolution_86, squeeze_262, relu_87, cat_55, squeeze_265, relu_88, convolution_88, squeeze_268, relu_89, convolution_89, squeeze_271, relu_90, cat_57, squeeze_274, relu_91, convolution_91, squeeze_277, relu_92, convolution_92, squeeze_280, relu_93, cat_59, squeeze_283, relu_94, convolution_94, squeeze_286, relu_95, convolution_95, squeeze_289, relu_96, cat_61, squeeze_292, relu_97, convolution_97, squeeze_295, relu_98, convolution_98, squeeze_298, relu_99, cat_63, squeeze_301, relu_100, relu_101, convolution_101, squeeze_307, relu_102, convolution_102, squeeze_310, relu_103, cat_65, squeeze_313, relu_104, convolution_104, squeeze_316, relu_105, convolution_105, squeeze_319, relu_106, cat_67, squeeze_322, relu_107, convolution_107, squeeze_325, relu_108, convolution_108, squeeze_328, relu_109, cat_69, squeeze_331, mean, le, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, unsqueeze_1022, unsqueeze_1034, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, unsqueeze_1094, unsqueeze_1106, unsqueeze_1118, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1382, unsqueeze_1394, unsqueeze_1406, unsqueeze_1418, unsqueeze_1430, unsqueeze_1442, unsqueeze_1454, unsqueeze_1466, unsqueeze_1478, unsqueeze_1490, unsqueeze_1502, unsqueeze_1514, unsqueeze_1526, unsqueeze_1538, unsqueeze_1550, unsqueeze_1562, unsqueeze_1574, unsqueeze_1586, unsqueeze_1610, unsqueeze_1622, unsqueeze_1634, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, unsqueeze_1682, unsqueeze_1694, unsqueeze_1706, unsqueeze_1718, unsqueeze_1730, sub_543, unsqueeze_1766, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (200, ), (1, ))
    assert_size_stride(primals_9, (200, ), (1, ))
    assert_size_stride(primals_11, (316, ), (1, ))
    assert_size_stride(primals_13, (200, ), (1, ))
    assert_size_stride(primals_15, (200, ), (1, ))
    assert_size_stride(primals_17, (336, ), (1, ))
    assert_size_stride(primals_19, (200, ), (1, ))
    assert_size_stride(primals_21, (200, ), (1, ))
    assert_size_stride(primals_23, (356, ), (1, ))
    assert_size_stride(primals_25, (200, ), (1, ))
    assert_size_stride(primals_27, (200, ), (1, ))
    assert_size_stride(primals_29, (376, ), (1, ))
    assert_size_stride(primals_31, (376, ), (1, ))
    assert_size_stride(primals_33, (400, ), (1, ))
    assert_size_stride(primals_35, (400, ), (1, ))
    assert_size_stride(primals_37, (704, ), (1, ))
    assert_size_stride(primals_39, (400, ), (1, ))
    assert_size_stride(primals_41, (400, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_45, (400, ), (1, ))
    assert_size_stride(primals_47, (400, ), (1, ))
    assert_size_stride(primals_49, (832, ), (1, ))
    assert_size_stride(primals_51, (400, ), (1, ))
    assert_size_stride(primals_53, (400, ), (1, ))
    assert_size_stride(primals_55, (896, ), (1, ))
    assert_size_stride(primals_57, (400, ), (1, ))
    assert_size_stride(primals_59, (400, ), (1, ))
    assert_size_stride(primals_61, (960, ), (1, ))
    assert_size_stride(primals_63, (400, ), (1, ))
    assert_size_stride(primals_65, (400, ), (1, ))
    assert_size_stride(primals_67, (1024, ), (1, ))
    assert_size_stride(primals_69, (400, ), (1, ))
    assert_size_stride(primals_71, (400, ), (1, ))
    assert_size_stride(primals_73, (1088, ), (1, ))
    assert_size_stride(primals_75, (400, ), (1, ))
    assert_size_stride(primals_77, (400, ), (1, ))
    assert_size_stride(primals_79, (1152, ), (1, ))
    assert_size_stride(primals_81, (1152, ), (1, ))
    assert_size_stride(primals_83, (800, ), (1, ))
    assert_size_stride(primals_85, (800, ), (1, ))
    assert_size_stride(primals_87, (1216, ), (1, ))
    assert_size_stride(primals_89, (800, ), (1, ))
    assert_size_stride(primals_91, (800, ), (1, ))
    assert_size_stride(primals_93, (1280, ), (1, ))
    assert_size_stride(primals_95, (800, ), (1, ))
    assert_size_stride(primals_97, (800, ), (1, ))
    assert_size_stride(primals_99, (1344, ), (1, ))
    assert_size_stride(primals_101, (800, ), (1, ))
    assert_size_stride(primals_103, (800, ), (1, ))
    assert_size_stride(primals_105, (1408, ), (1, ))
    assert_size_stride(primals_107, (800, ), (1, ))
    assert_size_stride(primals_109, (800, ), (1, ))
    assert_size_stride(primals_111, (1472, ), (1, ))
    assert_size_stride(primals_113, (800, ), (1, ))
    assert_size_stride(primals_115, (800, ), (1, ))
    assert_size_stride(primals_117, (1536, ), (1, ))
    assert_size_stride(primals_119, (800, ), (1, ))
    assert_size_stride(primals_121, (800, ), (1, ))
    assert_size_stride(primals_123, (1600, ), (1, ))
    assert_size_stride(primals_125, (800, ), (1, ))
    assert_size_stride(primals_127, (800, ), (1, ))
    assert_size_stride(primals_129, (1664, ), (1, ))
    assert_size_stride(primals_131, (800, ), (1, ))
    assert_size_stride(primals_133, (800, ), (1, ))
    assert_size_stride(primals_135, (1728, ), (1, ))
    assert_size_stride(primals_137, (800, ), (1, ))
    assert_size_stride(primals_139, (800, ), (1, ))
    assert_size_stride(primals_141, (1792, ), (1, ))
    assert_size_stride(primals_143, (800, ), (1, ))
    assert_size_stride(primals_145, (800, ), (1, ))
    assert_size_stride(primals_147, (1856, ), (1, ))
    assert_size_stride(primals_149, (800, ), (1, ))
    assert_size_stride(primals_151, (800, ), (1, ))
    assert_size_stride(primals_153, (1920, ), (1, ))
    assert_size_stride(primals_155, (800, ), (1, ))
    assert_size_stride(primals_157, (800, ), (1, ))
    assert_size_stride(primals_159, (1984, ), (1, ))
    assert_size_stride(primals_161, (800, ), (1, ))
    assert_size_stride(primals_163, (800, ), (1, ))
    assert_size_stride(primals_165, (2048, ), (1, ))
    assert_size_stride(primals_167, (800, ), (1, ))
    assert_size_stride(primals_169, (800, ), (1, ))
    assert_size_stride(primals_171, (2112, ), (1, ))
    assert_size_stride(primals_173, (800, ), (1, ))
    assert_size_stride(primals_175, (800, ), (1, ))
    assert_size_stride(primals_177, (2176, ), (1, ))
    assert_size_stride(primals_179, (800, ), (1, ))
    assert_size_stride(primals_181, (800, ), (1, ))
    assert_size_stride(primals_183, (2240, ), (1, ))
    assert_size_stride(primals_185, (800, ), (1, ))
    assert_size_stride(primals_187, (800, ), (1, ))
    assert_size_stride(primals_189, (2304, ), (1, ))
    assert_size_stride(primals_191, (800, ), (1, ))
    assert_size_stride(primals_193, (800, ), (1, ))
    assert_size_stride(primals_195, (2368, ), (1, ))
    assert_size_stride(primals_197, (800, ), (1, ))
    assert_size_stride(primals_199, (800, ), (1, ))
    assert_size_stride(primals_201, (2432, ), (1, ))
    assert_size_stride(primals_203, (2432, ), (1, ))
    assert_size_stride(primals_205, (1600, ), (1, ))
    assert_size_stride(primals_207, (1600, ), (1, ))
    assert_size_stride(primals_209, (2432, ), (1, ))
    assert_size_stride(primals_211, (1600, ), (1, ))
    assert_size_stride(primals_213, (1600, ), (1, ))
    assert_size_stride(primals_215, (2560, ), (1, ))
    assert_size_stride(primals_217, (1600, ), (1, ))
    assert_size_stride(primals_219, (1600, ), (1, ))
    assert_size_stride(primals_221, (2688, ), (1, ))
    assert_size_stride(primals_223, (128, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_224, (296, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_225, (200, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_226, (200, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_227, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_228, (200, 316, 1, 1), (316, 1, 1, 1))
    assert_size_stride(primals_229, (200, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_230, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_231, (200, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_232, (200, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_233, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_234, (200, 356, 1, 1), (356, 1, 1, 1))
    assert_size_stride(primals_235, (200, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_236, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_237, (640, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(primals_238, (400, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(primals_239, (400, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_240, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_241, (400, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_242, (400, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_243, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_244, (400, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_245, (400, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_246, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_247, (400, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_248, (400, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_249, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_250, (400, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_251, (400, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_252, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_253, (400, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_254, (400, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_255, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_256, (400, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_257, (400, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_258, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_259, (400, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_260, (400, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_261, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_262, (1152, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_263, (800, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_264, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_265, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_266, (800, 1216, 1, 1), (1216, 1, 1, 1))
    assert_size_stride(primals_267, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_268, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_269, (800, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_270, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_271, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_272, (800, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_273, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_274, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_275, (800, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(primals_276, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_277, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_278, (800, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(primals_279, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_280, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_281, (800, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_282, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_283, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_284, (800, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_285, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_286, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_287, (800, 1664, 1, 1), (1664, 1, 1, 1))
    assert_size_stride(primals_288, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_289, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_290, (800, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(primals_291, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_292, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_293, (800, 1792, 1, 1), (1792, 1, 1, 1))
    assert_size_stride(primals_294, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_295, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_296, (800, 1856, 1, 1), (1856, 1, 1, 1))
    assert_size_stride(primals_297, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_298, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_299, (800, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_300, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_301, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_302, (800, 1984, 1, 1), (1984, 1, 1, 1))
    assert_size_stride(primals_303, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_304, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_305, (800, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_306, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_307, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_308, (800, 2112, 1, 1), (2112, 1, 1, 1))
    assert_size_stride(primals_309, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_310, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_311, (800, 2176, 1, 1), (2176, 1, 1, 1))
    assert_size_stride(primals_312, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_313, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_314, (800, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(primals_315, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_316, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_317, (800, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_318, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_319, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_320, (800, 2368, 1, 1), (2368, 1, 1, 1))
    assert_size_stride(primals_321, (800, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_322, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_323, (2304, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_324, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_325, (1600, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_326, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_327, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_328, (1600, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_329, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_330, (1600, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(primals_331, (1600, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_332, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_333, (1000, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(primals_668, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 128, 112, 112), (1605632, 1, 14336, 128))
    assert_size_stride(squeeze_1, (128, ), (1, ))
    assert_size_stride(relu, (8, 128, 112, 112), (1605632, 1, 14336, 128))
    assert_size_stride(getitem_3, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_4, (128, ), (1, ))
    assert_size_stride(relu_1, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_2, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_2, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(squeeze_10, (200, ), (1, ))
    assert_size_stride(relu_3, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(convolution_3, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(squeeze_13, (200, ), (1, ))
    assert_size_stride(relu_4, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(cat_1, (8, 316, 56, 56), (990976, 1, 17696, 316))
    assert_size_stride(squeeze_16, (316, ), (1, ))
    assert_size_stride(relu_5, (8, 316, 56, 56), (990976, 1, 17696, 316))
    assert_size_stride(convolution_5, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(squeeze_19, (200, ), (1, ))
    assert_size_stride(relu_6, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(convolution_6, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(squeeze_22, (200, ), (1, ))
    assert_size_stride(relu_7, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(cat_3, (8, 336, 56, 56), (1053696, 1, 18816, 336))
    assert_size_stride(squeeze_25, (336, ), (1, ))
    assert_size_stride(relu_8, (8, 336, 56, 56), (1053696, 1, 18816, 336))
    assert_size_stride(convolution_8, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(squeeze_28, (200, ), (1, ))
    assert_size_stride(relu_9, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(convolution_9, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(squeeze_31, (200, ), (1, ))
    assert_size_stride(relu_10, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(cat_5, (8, 356, 56, 56), (1116416, 1, 19936, 356))
    assert_size_stride(squeeze_34, (356, ), (1, ))
    assert_size_stride(relu_11, (8, 356, 56, 56), (1116416, 1, 19936, 356))
    assert_size_stride(convolution_11, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(squeeze_37, (200, ), (1, ))
    assert_size_stride(relu_12, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(convolution_12, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(squeeze_40, (200, ), (1, ))
    assert_size_stride(relu_13, (8, 200, 56, 56), (627200, 1, 11200, 200))
    assert_size_stride(cat_7, (8, 376, 56, 56), (1179136, 1, 21056, 376))
    assert_size_stride(squeeze_43, (376, ), (1, ))
    assert_size_stride(relu_14, (8, 376, 56, 56), (1179136, 1, 21056, 376))
    assert_size_stride(relu_15, (8, 376, 56, 56), (1179136, 1, 21056, 376))
    assert_size_stride(convolution_15, (8, 400, 56, 56), (1254400, 1, 22400, 400))
    assert_size_stride(squeeze_49, (400, ), (1, ))
    assert_size_stride(relu_16, (8, 400, 56, 56), (1254400, 1, 22400, 400))
    assert_size_stride(convolution_16, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_52, (400, ), (1, ))
    assert_size_stride(relu_17, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(cat_9, (8, 704, 28, 28), (551936, 1, 19712, 704))
    assert_size_stride(squeeze_55, (704, ), (1, ))
    assert_size_stride(relu_18, (8, 704, 28, 28), (551936, 1, 19712, 704))
    assert_size_stride(convolution_18, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_58, (400, ), (1, ))
    assert_size_stride(relu_19, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(convolution_19, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_61, (400, ), (1, ))
    assert_size_stride(relu_20, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(cat_11, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_64, (768, ), (1, ))
    assert_size_stride(relu_21, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_21, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_67, (400, ), (1, ))
    assert_size_stride(relu_22, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(convolution_22, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_70, (400, ), (1, ))
    assert_size_stride(relu_23, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(cat_13, (8, 832, 28, 28), (652288, 1, 23296, 832))
    assert_size_stride(squeeze_73, (832, ), (1, ))
    assert_size_stride(relu_24, (8, 832, 28, 28), (652288, 1, 23296, 832))
    assert_size_stride(convolution_24, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_76, (400, ), (1, ))
    assert_size_stride(relu_25, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(convolution_25, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_79, (400, ), (1, ))
    assert_size_stride(relu_26, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(cat_15, (8, 896, 28, 28), (702464, 1, 25088, 896))
    assert_size_stride(squeeze_82, (896, ), (1, ))
    assert_size_stride(relu_27, (8, 896, 28, 28), (702464, 1, 25088, 896))
    assert_size_stride(convolution_27, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_85, (400, ), (1, ))
    assert_size_stride(relu_28, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(convolution_28, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_88, (400, ), (1, ))
    assert_size_stride(relu_29, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(cat_17, (8, 960, 28, 28), (752640, 1, 26880, 960))
    assert_size_stride(squeeze_91, (960, ), (1, ))
    assert_size_stride(relu_30, (8, 960, 28, 28), (752640, 1, 26880, 960))
    assert_size_stride(convolution_30, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_94, (400, ), (1, ))
    assert_size_stride(relu_31, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(convolution_31, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_97, (400, ), (1, ))
    assert_size_stride(relu_32, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(cat_19, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
    assert_size_stride(squeeze_100, (1024, ), (1, ))
    assert_size_stride(relu_33, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
    assert_size_stride(convolution_33, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_103, (400, ), (1, ))
    assert_size_stride(relu_34, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(convolution_34, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_106, (400, ), (1, ))
    assert_size_stride(relu_35, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(cat_21, (8, 1088, 28, 28), (852992, 1, 30464, 1088))
    assert_size_stride(squeeze_109, (1088, ), (1, ))
    assert_size_stride(relu_36, (8, 1088, 28, 28), (852992, 1, 30464, 1088))
    assert_size_stride(convolution_36, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_112, (400, ), (1, ))
    assert_size_stride(relu_37, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(convolution_37, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(squeeze_115, (400, ), (1, ))
    assert_size_stride(relu_38, (8, 400, 28, 28), (313600, 1, 11200, 400))
    assert_size_stride(cat_23, (8, 1152, 28, 28), (903168, 1, 32256, 1152))
    assert_size_stride(squeeze_118, (1152, ), (1, ))
    assert_size_stride(relu_39, (8, 1152, 28, 28), (903168, 1, 32256, 1152))
    assert_size_stride(relu_40, (8, 1152, 28, 28), (903168, 1, 32256, 1152))
    assert_size_stride(convolution_40, (8, 800, 28, 28), (627200, 1, 22400, 800))
    assert_size_stride(squeeze_124, (800, ), (1, ))
    assert_size_stride(relu_41, (8, 800, 28, 28), (627200, 1, 22400, 800))
    assert_size_stride(convolution_41, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_127, (800, ), (1, ))
    assert_size_stride(relu_42, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_25, (8, 1216, 14, 14), (238336, 1, 17024, 1216))
    assert_size_stride(squeeze_130, (1216, ), (1, ))
    assert_size_stride(relu_43, (8, 1216, 14, 14), (238336, 1, 17024, 1216))
    assert_size_stride(convolution_43, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_133, (800, ), (1, ))
    assert_size_stride(relu_44, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_44, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_136, (800, ), (1, ))
    assert_size_stride(relu_45, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_27, (8, 1280, 14, 14), (250880, 1, 17920, 1280))
    assert_size_stride(squeeze_139, (1280, ), (1, ))
    assert_size_stride(relu_46, (8, 1280, 14, 14), (250880, 1, 17920, 1280))
    assert_size_stride(convolution_46, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_142, (800, ), (1, ))
    assert_size_stride(relu_47, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_47, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_145, (800, ), (1, ))
    assert_size_stride(relu_48, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_29, (8, 1344, 14, 14), (263424, 1, 18816, 1344))
    assert_size_stride(squeeze_148, (1344, ), (1, ))
    assert_size_stride(relu_49, (8, 1344, 14, 14), (263424, 1, 18816, 1344))
    assert_size_stride(convolution_49, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_151, (800, ), (1, ))
    assert_size_stride(relu_50, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_50, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_154, (800, ), (1, ))
    assert_size_stride(relu_51, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_31, (8, 1408, 14, 14), (275968, 1, 19712, 1408))
    assert_size_stride(squeeze_157, (1408, ), (1, ))
    assert_size_stride(relu_52, (8, 1408, 14, 14), (275968, 1, 19712, 1408))
    assert_size_stride(convolution_52, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_160, (800, ), (1, ))
    assert_size_stride(relu_53, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_53, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_163, (800, ), (1, ))
    assert_size_stride(relu_54, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_33, (8, 1472, 14, 14), (288512, 1, 20608, 1472))
    assert_size_stride(squeeze_166, (1472, ), (1, ))
    assert_size_stride(relu_55, (8, 1472, 14, 14), (288512, 1, 20608, 1472))
    assert_size_stride(convolution_55, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_169, (800, ), (1, ))
    assert_size_stride(relu_56, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_56, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_172, (800, ), (1, ))
    assert_size_stride(relu_57, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_35, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(squeeze_175, (1536, ), (1, ))
    assert_size_stride(relu_58, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_58, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_178, (800, ), (1, ))
    assert_size_stride(relu_59, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_59, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_181, (800, ), (1, ))
    assert_size_stride(relu_60, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_37, (8, 1600, 14, 14), (313600, 1, 22400, 1600))
    assert_size_stride(squeeze_184, (1600, ), (1, ))
    assert_size_stride(relu_61, (8, 1600, 14, 14), (313600, 1, 22400, 1600))
    assert_size_stride(convolution_61, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_187, (800, ), (1, ))
    assert_size_stride(relu_62, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_62, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_190, (800, ), (1, ))
    assert_size_stride(relu_63, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_39, (8, 1664, 14, 14), (326144, 1, 23296, 1664))
    assert_size_stride(squeeze_193, (1664, ), (1, ))
    assert_size_stride(relu_64, (8, 1664, 14, 14), (326144, 1, 23296, 1664))
    assert_size_stride(convolution_64, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_196, (800, ), (1, ))
    assert_size_stride(relu_65, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_65, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_199, (800, ), (1, ))
    assert_size_stride(relu_66, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_41, (8, 1728, 14, 14), (338688, 1, 24192, 1728))
    assert_size_stride(squeeze_202, (1728, ), (1, ))
    assert_size_stride(relu_67, (8, 1728, 14, 14), (338688, 1, 24192, 1728))
    assert_size_stride(convolution_67, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_205, (800, ), (1, ))
    assert_size_stride(relu_68, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_68, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_208, (800, ), (1, ))
    assert_size_stride(relu_69, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_43, (8, 1792, 14, 14), (351232, 1, 25088, 1792))
    assert_size_stride(squeeze_211, (1792, ), (1, ))
    assert_size_stride(relu_70, (8, 1792, 14, 14), (351232, 1, 25088, 1792))
    assert_size_stride(convolution_70, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_214, (800, ), (1, ))
    assert_size_stride(relu_71, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_71, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_217, (800, ), (1, ))
    assert_size_stride(relu_72, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_45, (8, 1856, 14, 14), (363776, 1, 25984, 1856))
    assert_size_stride(squeeze_220, (1856, ), (1, ))
    assert_size_stride(relu_73, (8, 1856, 14, 14), (363776, 1, 25984, 1856))
    assert_size_stride(convolution_73, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_223, (800, ), (1, ))
    assert_size_stride(relu_74, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_74, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_226, (800, ), (1, ))
    assert_size_stride(relu_75, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_47, (8, 1920, 14, 14), (376320, 1, 26880, 1920))
    assert_size_stride(squeeze_229, (1920, ), (1, ))
    assert_size_stride(relu_76, (8, 1920, 14, 14), (376320, 1, 26880, 1920))
    assert_size_stride(convolution_76, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_232, (800, ), (1, ))
    assert_size_stride(relu_77, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_77, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_235, (800, ), (1, ))
    assert_size_stride(relu_78, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_49, (8, 1984, 14, 14), (388864, 1, 27776, 1984))
    assert_size_stride(squeeze_238, (1984, ), (1, ))
    assert_size_stride(relu_79, (8, 1984, 14, 14), (388864, 1, 27776, 1984))
    assert_size_stride(convolution_79, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_241, (800, ), (1, ))
    assert_size_stride(relu_80, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_80, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_244, (800, ), (1, ))
    assert_size_stride(relu_81, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_51, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    assert_size_stride(squeeze_247, (2048, ), (1, ))
    assert_size_stride(relu_82, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    assert_size_stride(convolution_82, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_250, (800, ), (1, ))
    assert_size_stride(relu_83, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_83, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_253, (800, ), (1, ))
    assert_size_stride(relu_84, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_53, (8, 2112, 14, 14), (413952, 1, 29568, 2112))
    assert_size_stride(squeeze_256, (2112, ), (1, ))
    assert_size_stride(relu_85, (8, 2112, 14, 14), (413952, 1, 29568, 2112))
    assert_size_stride(convolution_85, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_259, (800, ), (1, ))
    assert_size_stride(relu_86, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_86, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_262, (800, ), (1, ))
    assert_size_stride(relu_87, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_55, (8, 2176, 14, 14), (426496, 1, 30464, 2176))
    assert_size_stride(squeeze_265, (2176, ), (1, ))
    assert_size_stride(relu_88, (8, 2176, 14, 14), (426496, 1, 30464, 2176))
    assert_size_stride(convolution_88, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_268, (800, ), (1, ))
    assert_size_stride(relu_89, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_89, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_271, (800, ), (1, ))
    assert_size_stride(relu_90, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_57, (8, 2240, 14, 14), (439040, 1, 31360, 2240))
    assert_size_stride(squeeze_274, (2240, ), (1, ))
    assert_size_stride(relu_91, (8, 2240, 14, 14), (439040, 1, 31360, 2240))
    assert_size_stride(convolution_91, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_277, (800, ), (1, ))
    assert_size_stride(relu_92, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_92, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_280, (800, ), (1, ))
    assert_size_stride(relu_93, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_59, (8, 2304, 14, 14), (451584, 1, 32256, 2304))
    assert_size_stride(squeeze_283, (2304, ), (1, ))
    assert_size_stride(relu_94, (8, 2304, 14, 14), (451584, 1, 32256, 2304))
    assert_size_stride(convolution_94, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_286, (800, ), (1, ))
    assert_size_stride(relu_95, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_95, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_289, (800, ), (1, ))
    assert_size_stride(relu_96, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_61, (8, 2368, 14, 14), (464128, 1, 33152, 2368))
    assert_size_stride(squeeze_292, (2368, ), (1, ))
    assert_size_stride(relu_97, (8, 2368, 14, 14), (464128, 1, 33152, 2368))
    assert_size_stride(convolution_97, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_295, (800, ), (1, ))
    assert_size_stride(relu_98, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(convolution_98, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(squeeze_298, (800, ), (1, ))
    assert_size_stride(relu_99, (8, 800, 14, 14), (156800, 1, 11200, 800))
    assert_size_stride(cat_63, (8, 2432, 14, 14), (476672, 1, 34048, 2432))
    assert_size_stride(squeeze_301, (2432, ), (1, ))
    assert_size_stride(relu_100, (8, 2432, 14, 14), (476672, 1, 34048, 2432))
    assert_size_stride(relu_101, (8, 2432, 14, 14), (476672, 1, 34048, 2432))
    assert_size_stride(convolution_101, (8, 1600, 14, 14), (313600, 1, 22400, 1600))
    assert_size_stride(squeeze_307, (1600, ), (1, ))
    assert_size_stride(relu_102, (8, 1600, 14, 14), (313600, 1, 22400, 1600))
    assert_size_stride(convolution_102, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(squeeze_310, (1600, ), (1, ))
    assert_size_stride(relu_103, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(cat_65, (8, 2432, 7, 7), (119168, 1, 17024, 2432))
    assert_size_stride(squeeze_313, (2432, ), (1, ))
    assert_size_stride(relu_104, (8, 2432, 7, 7), (119168, 1, 17024, 2432))
    assert_size_stride(convolution_104, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(squeeze_316, (1600, ), (1, ))
    assert_size_stride(relu_105, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(convolution_105, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(squeeze_319, (1600, ), (1, ))
    assert_size_stride(relu_106, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(cat_67, (8, 2560, 7, 7), (125440, 1, 17920, 2560))
    assert_size_stride(squeeze_322, (2560, ), (1, ))
    assert_size_stride(relu_107, (8, 2560, 7, 7), (125440, 1, 17920, 2560))
    assert_size_stride(convolution_107, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(squeeze_325, (1600, ), (1, ))
    assert_size_stride(relu_108, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(convolution_108, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(squeeze_328, (1600, ), (1, ))
    assert_size_stride(relu_109, (8, 1600, 7, 7), (78400, 1, 11200, 1600))
    assert_size_stride(cat_69, (8, 2688, 7, 7), (131712, 1, 18816, 2688))
    assert_size_stride(squeeze_331, (2688, ), (1, ))
    assert_size_stride(mean, (8, 2688, 1, 1), (2688, 1, 2688, 2688))
    assert_size_stride(le, (8, 2688, 7, 7), (131712, 1, 18816, 2688))
    assert_size_stride(unsqueeze_446, (1, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_470, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(unsqueeze_494, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_518, (1, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(unsqueeze_530, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_542, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_554, (1, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(unsqueeze_578, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_590, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_602, (1, 2368, 1, 1), (2368, 1, 1, 1))
    assert_size_stride(unsqueeze_614, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_626, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_638, (1, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(unsqueeze_650, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_662, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_674, (1, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(unsqueeze_686, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_698, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_710, (1, 2176, 1, 1), (2176, 1, 1, 1))
    assert_size_stride(unsqueeze_722, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_734, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_746, (1, 2112, 1, 1), (2112, 1, 1, 1))
    assert_size_stride(unsqueeze_758, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_770, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_782, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_794, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_806, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_818, (1, 1984, 1, 1), (1984, 1, 1, 1))
    assert_size_stride(unsqueeze_830, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_842, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_854, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_866, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_878, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_890, (1, 1856, 1, 1), (1856, 1, 1, 1))
    assert_size_stride(unsqueeze_902, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_914, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_926, (1, 1792, 1, 1), (1792, 1, 1, 1))
    assert_size_stride(unsqueeze_938, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_950, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_962, (1, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(unsqueeze_974, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_986, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_998, (1, 1664, 1, 1), (1664, 1, 1, 1))
    assert_size_stride(unsqueeze_1010, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1022, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1034, (1, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(unsqueeze_1046, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1058, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1070, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(unsqueeze_1082, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1094, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1106, (1, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(unsqueeze_1118, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1130, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1142, (1, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(unsqueeze_1154, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1166, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1178, (1, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(unsqueeze_1190, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1202, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1214, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_1226, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1238, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1250, (1, 1216, 1, 1), (1216, 1, 1, 1))
    assert_size_stride(unsqueeze_1262, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1274, (1, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(unsqueeze_1286, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_1310, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1322, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1334, (1, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(unsqueeze_1346, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1358, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1370, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_1382, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1394, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1406, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_1418, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1430, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1442, (1, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(unsqueeze_1454, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1466, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1478, (1, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(unsqueeze_1490, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1502, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1514, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_1526, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1538, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1550, (1, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(unsqueeze_1562, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1574, (1, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(unsqueeze_1586, (1, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(unsqueeze_1610, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1622, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1634, (1, 356, 1, 1), (356, 1, 1, 1))
    assert_size_stride(unsqueeze_1646, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1658, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1670, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_1682, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1694, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1706, (1, 316, 1, 1), (316, 1, 1, 1))
    assert_size_stride(unsqueeze_1718, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(unsqueeze_1730, (1, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(sub_543, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(unsqueeze_1766, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf0 = aten.convolution_backward(reinterpret_tensor(tangents_1, (8, 1000, 1, 1), (1000, 1, 1, 1), 0), mean, primals_333, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean
    del primals_333
    del tangents_1
    buf1 = buf0[0]
    buf2 = buf0[1]
    buf3 = buf0[2]
    del buf0
    buf4 = empty((2688, ), device='cpu', dtype=torch.float32)
    buf5 = empty((2688, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 2688, 7, 7), (131712, 1, 18816, 2688), device='cpu', dtype=torch.float32)
    buf7 = buf5; del buf5  # reuse
    buf8 = empty((8, 2176, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_slice_backward_threshold_backward_0(c_void_p(buf7.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(cat_69.data_ptr()), c_void_p(unsqueeze_446.data_ptr()), c_void_p(squeeze_331.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()))
    del buf1
    del cat_69
    del le
    del primals_221
    del squeeze_331
    del unsqueeze_446
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf9 = aten.convolution_backward(buf8, relu_109, primals_332, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_332
    buf10 = buf9[0]
    buf11 = buf9[1]
    del buf9
    buf12 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf13 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf14 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf15 = buf10; del buf10  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1(c_void_p(buf15.data_ptr()), c_void_p(relu_109.data_ptr()), c_void_p(convolution_108.data_ptr()), c_void_p(unsqueeze_458.data_ptr()), c_void_p(squeeze_328.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()))
    del convolution_108
    del primals_219
    del relu_109
    del squeeze_328
    del unsqueeze_458
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf16 = aten.convolution_backward(buf15, relu_108, primals_331, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf15
    del primals_331
    buf17 = buf16[0]
    buf18 = buf16[1]
    del buf16
    buf19 = buf13; del buf13  # reuse
    buf20 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf21 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf22 = buf17; del buf17  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2(c_void_p(buf22.data_ptr()), c_void_p(relu_108.data_ptr()), c_void_p(convolution_107.data_ptr()), c_void_p(unsqueeze_470.data_ptr()), c_void_p(squeeze_325.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del convolution_107
    del primals_217
    del relu_108
    del squeeze_325
    del unsqueeze_470
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf23 = aten.convolution_backward(buf22, relu_107, primals_330, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf22
    del primals_330
    buf24 = buf23[0]
    buf25 = buf23[1]
    del buf23
    buf26 = empty((2560, ), device='cpu', dtype=torch.float32)
    buf27 = empty((2560, ), device='cpu', dtype=torch.float32)
    buf28 = buf24; del buf24  # reuse
    buf29 = buf27; del buf27  # reuse
    buf30 = buf8; del buf8  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_3(c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(relu_107.data_ptr()), c_void_p(cat_67.data_ptr()), c_void_p(unsqueeze_482.data_ptr()), c_void_p(squeeze_322.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf30.data_ptr()))
    del cat_67
    del primals_215
    del relu_107
    del squeeze_322
    del unsqueeze_482
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf31 = aten.convolution_backward(buf30, relu_106, primals_329, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_329
    buf32 = buf31[0]
    buf33 = buf31[1]
    del buf31
    buf34 = buf20; del buf20  # reuse
    buf35 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf36 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf37 = buf32; del buf32  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf37.data_ptr()), c_void_p(relu_106.data_ptr()), c_void_p(convolution_105.data_ptr()), c_void_p(unsqueeze_494.data_ptr()), c_void_p(squeeze_319.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del convolution_105
    del primals_213
    del relu_106
    del squeeze_319
    del unsqueeze_494
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf38 = aten.convolution_backward(buf37, relu_105, primals_328, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf37
    del primals_328
    buf39 = buf38[0]
    buf40 = buf38[1]
    del buf38
    buf41 = buf35; del buf35  # reuse
    buf42 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf43 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf44 = buf39; del buf39  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(buf44.data_ptr()), c_void_p(relu_105.data_ptr()), c_void_p(convolution_104.data_ptr()), c_void_p(unsqueeze_506.data_ptr()), c_void_p(squeeze_316.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del convolution_104
    del primals_211
    del relu_105
    del squeeze_316
    del unsqueeze_506
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf45 = aten.convolution_backward(buf44, relu_104, primals_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf44
    del primals_327
    buf46 = buf45[0]
    buf47 = buf45[1]
    del buf45
    buf48 = empty((2432, ), device='cpu', dtype=torch.float32)
    buf49 = empty((2432, ), device='cpu', dtype=torch.float32)
    buf50 = buf46; del buf46  # reuse
    buf51 = buf49; del buf49  # reuse
    buf52 = buf30; del buf30  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_6(c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(relu_104.data_ptr()), c_void_p(cat_65.data_ptr()), c_void_p(unsqueeze_518.data_ptr()), c_void_p(squeeze_313.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf52.data_ptr()))
    del cat_65
    del primals_209
    del relu_104
    del squeeze_313
    del unsqueeze_518
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf53 = aten.convolution_backward(buf52, relu_103, primals_326, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf52
    del primals_326
    buf54 = buf53[0]
    buf55 = buf53[1]
    del buf53
    buf56 = buf42; del buf42  # reuse
    buf57 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf58 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf59 = buf54; del buf54  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7(c_void_p(buf59.data_ptr()), c_void_p(relu_103.data_ptr()), c_void_p(convolution_102.data_ptr()), c_void_p(unsqueeze_530.data_ptr()), c_void_p(squeeze_310.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    del convolution_102
    del primals_207
    del relu_103
    del squeeze_310
    del unsqueeze_530
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf60 = aten.convolution_backward(buf59, relu_102, primals_325, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf59
    del primals_325
    buf61 = buf60[0]
    buf62 = buf60[1]
    del buf60
    buf63 = buf57; del buf57  # reuse
    buf64 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf65 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf66 = buf61; del buf61  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8(c_void_p(buf66.data_ptr()), c_void_p(relu_102.data_ptr()), c_void_p(convolution_101.data_ptr()), c_void_p(unsqueeze_542.data_ptr()), c_void_p(squeeze_307.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()))
    del convolution_101
    del primals_205
    del relu_102
    del squeeze_307
    del unsqueeze_542
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf67 = aten.convolution_backward(buf66, relu_101, primals_324, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf66
    del primals_324
    buf68 = buf67[0]
    buf69 = buf67[1]
    del buf67
    buf73 = empty((8, 2304, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_slice_backward_9(c_void_p(buf6.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf73.data_ptr()))
    del buf50
    del buf6
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf74 = aten.convolution_backward(buf73, relu_100, primals_323, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf73
    del primals_323
    buf75 = buf74[0]
    buf70 = empty((2432, ), device='cpu', dtype=torch.float32)
    buf71 = empty((2432, ), device='cpu', dtype=torch.float32)
    buf77 = empty((2432, ), device='cpu', dtype=torch.float32)
    buf78 = empty((2432, ), device='cpu', dtype=torch.float32)
    buf72 = empty((2432, ), device='cpu', dtype=torch.float32)
    buf79 = empty((2432, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_threshold_backward_10(c_void_p(relu_101.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(cat_63.data_ptr()), c_void_p(unsqueeze_554.data_ptr()), c_void_p(relu_100.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(squeeze_301.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf79.data_ptr()))
    buf76 = buf74[1]
    del buf74
    buf80 = buf68; del buf68  # reuse
    buf81 = empty((8, 1088, 14, 14), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_11(c_void_p(buf80.data_ptr()), c_void_p(relu_101.data_ptr()), c_void_p(cat_63.data_ptr()), c_void_p(unsqueeze_554.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(squeeze_301.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(relu_100.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(buf81.data_ptr()))
    del buf71
    del buf75
    del buf78
    del cat_63
    del primals_201
    del primals_203
    del relu_100
    del relu_101
    del squeeze_301
    del unsqueeze_554
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf82 = aten.convolution_backward(buf81, relu_99, primals_322, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_322
    buf83 = buf82[0]
    buf84 = buf82[1]
    del buf82
    buf85 = empty((800, ), device='cpu', dtype=torch.float32)
    buf86 = empty((800, ), device='cpu', dtype=torch.float32)
    buf87 = empty((800, ), device='cpu', dtype=torch.float32)
    buf88 = buf83; del buf83  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf88.data_ptr()), c_void_p(relu_99.data_ptr()), c_void_p(convolution_98.data_ptr()), c_void_p(unsqueeze_578.data_ptr()), c_void_p(squeeze_298.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    del convolution_98
    del primals_199
    del relu_99
    del squeeze_298
    del unsqueeze_578
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf89 = aten.convolution_backward(buf88, relu_98, primals_321, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf88
    del primals_321
    buf90 = buf89[0]
    buf91 = buf89[1]
    del buf89
    buf92 = buf86; del buf86  # reuse
    buf93 = empty((800, ), device='cpu', dtype=torch.float32)
    buf94 = empty((800, ), device='cpu', dtype=torch.float32)
    buf95 = buf90; del buf90  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(buf95.data_ptr()), c_void_p(relu_98.data_ptr()), c_void_p(convolution_97.data_ptr()), c_void_p(unsqueeze_590.data_ptr()), c_void_p(squeeze_295.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del convolution_97
    del primals_197
    del relu_98
    del squeeze_295
    del unsqueeze_590
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf96 = aten.convolution_backward(buf95, relu_97, primals_320, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf95
    del primals_320
    buf97 = buf96[0]
    buf98 = buf96[1]
    del buf96
    buf99 = empty((2368, ), device='cpu', dtype=torch.float32)
    buf100 = empty((2368, ), device='cpu', dtype=torch.float32)
    buf101 = buf97; del buf97  # reuse
    buf102 = buf100; del buf100  # reuse
    buf103 = buf81; del buf81  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_14(c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(relu_97.data_ptr()), c_void_p(cat_61.data_ptr()), c_void_p(unsqueeze_602.data_ptr()), c_void_p(squeeze_292.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf103.data_ptr()))
    del cat_61
    del primals_195
    del relu_97
    del squeeze_292
    del unsqueeze_602
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf104 = aten.convolution_backward(buf103, relu_96, primals_319, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_319
    buf105 = buf104[0]
    buf106 = buf104[1]
    del buf104
    buf107 = buf93; del buf93  # reuse
    buf108 = empty((800, ), device='cpu', dtype=torch.float32)
    buf109 = empty((800, ), device='cpu', dtype=torch.float32)
    buf110 = buf105; del buf105  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf110.data_ptr()), c_void_p(relu_96.data_ptr()), c_void_p(convolution_95.data_ptr()), c_void_p(unsqueeze_614.data_ptr()), c_void_p(squeeze_289.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del convolution_95
    del primals_193
    del relu_96
    del squeeze_289
    del unsqueeze_614
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf111 = aten.convolution_backward(buf110, relu_95, primals_318, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf110
    del primals_318
    buf112 = buf111[0]
    buf113 = buf111[1]
    del buf111
    buf114 = buf108; del buf108  # reuse
    buf115 = empty((800, ), device='cpu', dtype=torch.float32)
    buf116 = empty((800, ), device='cpu', dtype=torch.float32)
    buf117 = buf112; del buf112  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16(c_void_p(buf117.data_ptr()), c_void_p(relu_95.data_ptr()), c_void_p(convolution_94.data_ptr()), c_void_p(unsqueeze_626.data_ptr()), c_void_p(squeeze_286.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    del convolution_94
    del primals_191
    del relu_95
    del squeeze_286
    del unsqueeze_626
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf118 = aten.convolution_backward(buf117, relu_94, primals_317, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf117
    del primals_317
    buf119 = buf118[0]
    buf120 = buf118[1]
    del buf118
    buf121 = empty((2304, ), device='cpu', dtype=torch.float32)
    buf122 = empty((2304, ), device='cpu', dtype=torch.float32)
    buf123 = buf119; del buf119  # reuse
    buf124 = buf122; del buf122  # reuse
    buf125 = buf103; del buf103  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_17(c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(relu_94.data_ptr()), c_void_p(cat_59.data_ptr()), c_void_p(unsqueeze_638.data_ptr()), c_void_p(squeeze_283.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf125.data_ptr()))
    del cat_59
    del primals_189
    del relu_94
    del squeeze_283
    del unsqueeze_638
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf126 = aten.convolution_backward(buf125, relu_93, primals_316, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_316
    buf127 = buf126[0]
    buf128 = buf126[1]
    del buf126
    buf129 = buf115; del buf115  # reuse
    buf130 = empty((800, ), device='cpu', dtype=torch.float32)
    buf131 = empty((800, ), device='cpu', dtype=torch.float32)
    buf132 = buf127; del buf127  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(buf132.data_ptr()), c_void_p(relu_93.data_ptr()), c_void_p(convolution_92.data_ptr()), c_void_p(unsqueeze_650.data_ptr()), c_void_p(squeeze_280.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del convolution_92
    del primals_187
    del relu_93
    del squeeze_280
    del unsqueeze_650
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf133 = aten.convolution_backward(buf132, relu_92, primals_315, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf132
    del primals_315
    buf134 = buf133[0]
    buf135 = buf133[1]
    del buf133
    buf136 = buf130; del buf130  # reuse
    buf137 = empty((800, ), device='cpu', dtype=torch.float32)
    buf138 = empty((800, ), device='cpu', dtype=torch.float32)
    buf139 = buf134; del buf134  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(buf139.data_ptr()), c_void_p(relu_92.data_ptr()), c_void_p(convolution_91.data_ptr()), c_void_p(unsqueeze_662.data_ptr()), c_void_p(squeeze_277.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    del convolution_91
    del primals_185
    del relu_92
    del squeeze_277
    del unsqueeze_662
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf140 = aten.convolution_backward(buf139, relu_91, primals_314, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf139
    del primals_314
    buf141 = buf140[0]
    buf142 = buf140[1]
    del buf140
    buf143 = empty((2240, ), device='cpu', dtype=torch.float32)
    buf144 = empty((2240, ), device='cpu', dtype=torch.float32)
    buf145 = buf141; del buf141  # reuse
    buf146 = buf144; del buf144  # reuse
    buf147 = buf125; del buf125  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_20(c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(relu_91.data_ptr()), c_void_p(cat_57.data_ptr()), c_void_p(unsqueeze_674.data_ptr()), c_void_p(squeeze_274.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf147.data_ptr()))
    del cat_57
    del primals_183
    del relu_91
    del squeeze_274
    del unsqueeze_674
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf148 = aten.convolution_backward(buf147, relu_90, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_313
    buf149 = buf148[0]
    buf150 = buf148[1]
    del buf148
    buf151 = buf137; del buf137  # reuse
    buf152 = empty((800, ), device='cpu', dtype=torch.float32)
    buf153 = empty((800, ), device='cpu', dtype=torch.float32)
    buf154 = buf149; del buf149  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21(c_void_p(buf154.data_ptr()), c_void_p(relu_90.data_ptr()), c_void_p(convolution_89.data_ptr()), c_void_p(unsqueeze_686.data_ptr()), c_void_p(squeeze_271.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    del convolution_89
    del primals_181
    del relu_90
    del squeeze_271
    del unsqueeze_686
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf155 = aten.convolution_backward(buf154, relu_89, primals_312, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf154
    del primals_312
    buf156 = buf155[0]
    buf157 = buf155[1]
    del buf155
    buf158 = buf152; del buf152  # reuse
    buf159 = empty((800, ), device='cpu', dtype=torch.float32)
    buf160 = empty((800, ), device='cpu', dtype=torch.float32)
    buf161 = buf156; del buf156  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(buf161.data_ptr()), c_void_p(relu_89.data_ptr()), c_void_p(convolution_88.data_ptr()), c_void_p(unsqueeze_698.data_ptr()), c_void_p(squeeze_268.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del convolution_88
    del primals_179
    del relu_89
    del squeeze_268
    del unsqueeze_698
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf162 = aten.convolution_backward(buf161, relu_88, primals_311, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf161
    del primals_311
    buf163 = buf162[0]
    buf164 = buf162[1]
    del buf162
    buf165 = empty((2176, ), device='cpu', dtype=torch.float32)
    buf166 = empty((2176, ), device='cpu', dtype=torch.float32)
    buf167 = buf163; del buf163  # reuse
    buf168 = buf166; del buf166  # reuse
    buf169 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    buf170 = empty_strided((8, 1152, 14, 14), (225792, 1, 16128, 1152), device='cpu', dtype=torch.float32)
    buf171 = buf147; del buf147  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_23(c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(relu_88.data_ptr()), c_void_p(cat_55.data_ptr()), c_void_p(unsqueeze_710.data_ptr()), c_void_p(squeeze_265.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    del buf101
    del buf145
    del buf167
    del buf80
    del cat_55
    del primals_177
    del relu_88
    del squeeze_265
    del unsqueeze_710
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf172 = aten.convolution_backward(buf171, relu_87, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_310
    buf173 = buf172[0]
    buf174 = buf172[1]
    del buf172
    buf175 = buf159; del buf159  # reuse
    buf176 = empty((800, ), device='cpu', dtype=torch.float32)
    buf177 = empty((800, ), device='cpu', dtype=torch.float32)
    buf178 = buf173; del buf173  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf178.data_ptr()), c_void_p(relu_87.data_ptr()), c_void_p(convolution_86.data_ptr()), c_void_p(unsqueeze_722.data_ptr()), c_void_p(squeeze_262.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    del convolution_86
    del primals_175
    del relu_87
    del squeeze_262
    del unsqueeze_722
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf179 = aten.convolution_backward(buf178, relu_86, primals_309, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf178
    del primals_309
    buf180 = buf179[0]
    buf181 = buf179[1]
    del buf179
    buf182 = buf176; del buf176  # reuse
    buf183 = empty((800, ), device='cpu', dtype=torch.float32)
    buf184 = empty((800, ), device='cpu', dtype=torch.float32)
    buf185 = buf180; del buf180  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25(c_void_p(buf185.data_ptr()), c_void_p(relu_86.data_ptr()), c_void_p(convolution_85.data_ptr()), c_void_p(unsqueeze_734.data_ptr()), c_void_p(squeeze_259.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    del convolution_85
    del primals_173
    del relu_86
    del squeeze_259
    del unsqueeze_734
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf186 = aten.convolution_backward(buf185, relu_85, primals_308, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf185
    del primals_308
    buf187 = buf186[0]
    buf188 = buf186[1]
    del buf186
    buf189 = empty((2112, ), device='cpu', dtype=torch.float32)
    buf190 = empty((2112, ), device='cpu', dtype=torch.float32)
    buf191 = buf187; del buf187  # reuse
    buf192 = buf190; del buf190  # reuse
    buf193 = buf171; del buf171  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_26(c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(relu_85.data_ptr()), c_void_p(cat_53.data_ptr()), c_void_p(unsqueeze_746.data_ptr()), c_void_p(squeeze_256.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf193.data_ptr()))
    del cat_53
    del primals_171
    del relu_85
    del squeeze_256
    del unsqueeze_746
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf194 = aten.convolution_backward(buf193, relu_84, primals_307, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_307
    buf195 = buf194[0]
    buf196 = buf194[1]
    del buf194
    buf197 = buf183; del buf183  # reuse
    buf198 = empty((800, ), device='cpu', dtype=torch.float32)
    buf199 = empty((800, ), device='cpu', dtype=torch.float32)
    buf200 = buf195; del buf195  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(buf200.data_ptr()), c_void_p(relu_84.data_ptr()), c_void_p(convolution_83.data_ptr()), c_void_p(unsqueeze_758.data_ptr()), c_void_p(squeeze_253.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    del convolution_83
    del primals_169
    del relu_84
    del squeeze_253
    del unsqueeze_758
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf201 = aten.convolution_backward(buf200, relu_83, primals_306, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf200
    del primals_306
    buf202 = buf201[0]
    buf203 = buf201[1]
    del buf201
    buf204 = buf198; del buf198  # reuse
    buf205 = empty((800, ), device='cpu', dtype=torch.float32)
    buf206 = empty((800, ), device='cpu', dtype=torch.float32)
    buf207 = buf202; del buf202  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28(c_void_p(buf207.data_ptr()), c_void_p(relu_83.data_ptr()), c_void_p(convolution_82.data_ptr()), c_void_p(unsqueeze_770.data_ptr()), c_void_p(squeeze_250.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del convolution_82
    del primals_167
    del relu_83
    del squeeze_250
    del unsqueeze_770
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf208 = aten.convolution_backward(buf207, relu_82, primals_305, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf207
    del primals_305
    buf209 = buf208[0]
    buf210 = buf208[1]
    del buf208
    buf211 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf212 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf213 = buf209; del buf209  # reuse
    buf214 = buf212; del buf212  # reuse
    buf215 = buf193; del buf193  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_29(c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(relu_82.data_ptr()), c_void_p(cat_51.data_ptr()), c_void_p(unsqueeze_782.data_ptr()), c_void_p(squeeze_247.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf215.data_ptr()))
    del cat_51
    del primals_165
    del relu_82
    del squeeze_247
    del unsqueeze_782
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf216 = aten.convolution_backward(buf215, relu_81, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_304
    buf217 = buf216[0]
    buf218 = buf216[1]
    del buf216
    buf219 = buf205; del buf205  # reuse
    buf220 = empty((800, ), device='cpu', dtype=torch.float32)
    buf221 = empty((800, ), device='cpu', dtype=torch.float32)
    buf222 = buf217; del buf217  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30(c_void_p(buf222.data_ptr()), c_void_p(relu_81.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(unsqueeze_794.data_ptr()), c_void_p(squeeze_244.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    del convolution_80
    del primals_163
    del relu_81
    del squeeze_244
    del unsqueeze_794
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf223 = aten.convolution_backward(buf222, relu_80, primals_303, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf222
    del primals_303
    buf224 = buf223[0]
    buf225 = buf223[1]
    del buf223
    buf226 = buf220; del buf220  # reuse
    buf227 = empty((800, ), device='cpu', dtype=torch.float32)
    buf228 = empty((800, ), device='cpu', dtype=torch.float32)
    buf229 = buf224; del buf224  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31(c_void_p(buf229.data_ptr()), c_void_p(relu_80.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(unsqueeze_806.data_ptr()), c_void_p(squeeze_241.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del convolution_79
    del primals_161
    del relu_80
    del squeeze_241
    del unsqueeze_806
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf230 = aten.convolution_backward(buf229, relu_79, primals_302, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf229
    del primals_302
    buf231 = buf230[0]
    buf232 = buf230[1]
    del buf230
    buf233 = empty((1984, ), device='cpu', dtype=torch.float32)
    buf234 = empty((1984, ), device='cpu', dtype=torch.float32)
    buf235 = buf231; del buf231  # reuse
    buf236 = buf234; del buf234  # reuse
    buf237 = buf215; del buf215  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_32(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(relu_79.data_ptr()), c_void_p(cat_49.data_ptr()), c_void_p(unsqueeze_818.data_ptr()), c_void_p(squeeze_238.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf237.data_ptr()))
    del cat_49
    del primals_159
    del relu_79
    del squeeze_238
    del unsqueeze_818
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf238 = aten.convolution_backward(buf237, relu_78, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_301
    buf239 = buf238[0]
    buf240 = buf238[1]
    del buf238
    buf241 = buf227; del buf227  # reuse
    buf242 = empty((800, ), device='cpu', dtype=torch.float32)
    buf243 = empty((800, ), device='cpu', dtype=torch.float32)
    buf244 = buf239; del buf239  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33(c_void_p(buf244.data_ptr()), c_void_p(relu_78.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(unsqueeze_830.data_ptr()), c_void_p(squeeze_235.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    del convolution_77
    del primals_157
    del relu_78
    del squeeze_235
    del unsqueeze_830
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf245 = aten.convolution_backward(buf244, relu_77, primals_300, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf244
    del primals_300
    buf246 = buf245[0]
    buf247 = buf245[1]
    del buf245
    buf248 = buf242; del buf242  # reuse
    buf249 = empty((800, ), device='cpu', dtype=torch.float32)
    buf250 = empty((800, ), device='cpu', dtype=torch.float32)
    buf251 = buf246; del buf246  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34(c_void_p(buf251.data_ptr()), c_void_p(relu_77.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(unsqueeze_842.data_ptr()), c_void_p(squeeze_232.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()))
    del convolution_76
    del primals_155
    del relu_77
    del squeeze_232
    del unsqueeze_842
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf252 = aten.convolution_backward(buf251, relu_76, primals_299, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf251
    del primals_299
    buf253 = buf252[0]
    buf254 = buf252[1]
    del buf252
    buf255 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf256 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf257 = buf253; del buf253  # reuse
    buf258 = buf256; del buf256  # reuse
    buf259 = buf169; del buf169  # reuse
    buf260 = empty_strided((8, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    buf261 = buf237; del buf237  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_35(c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(relu_76.data_ptr()), c_void_p(cat_47.data_ptr()), c_void_p(unsqueeze_854.data_ptr()), c_void_p(squeeze_229.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del buf191
    del buf235
    del buf257
    del cat_47
    del primals_153
    del relu_76
    del squeeze_229
    del unsqueeze_854
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf262 = aten.convolution_backward(buf261, relu_75, primals_298, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_298
    buf263 = buf262[0]
    buf264 = buf262[1]
    del buf262
    buf265 = buf249; del buf249  # reuse
    buf266 = empty((800, ), device='cpu', dtype=torch.float32)
    buf267 = empty((800, ), device='cpu', dtype=torch.float32)
    buf268 = buf263; del buf263  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf268.data_ptr()), c_void_p(relu_75.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(unsqueeze_866.data_ptr()), c_void_p(squeeze_226.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()))
    del convolution_74
    del primals_151
    del relu_75
    del squeeze_226
    del unsqueeze_866
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf269 = aten.convolution_backward(buf268, relu_74, primals_297, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf268
    del primals_297
    buf270 = buf269[0]
    buf271 = buf269[1]
    del buf269
    buf272 = buf266; del buf266  # reuse
    buf273 = empty((800, ), device='cpu', dtype=torch.float32)
    buf274 = empty((800, ), device='cpu', dtype=torch.float32)
    buf275 = buf270; del buf270  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37(c_void_p(buf275.data_ptr()), c_void_p(relu_74.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(unsqueeze_878.data_ptr()), c_void_p(squeeze_223.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del convolution_73
    del primals_149
    del relu_74
    del squeeze_223
    del unsqueeze_878
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf276 = aten.convolution_backward(buf275, relu_73, primals_296, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf275
    del primals_296
    buf277 = buf276[0]
    buf278 = buf276[1]
    del buf276
    buf279 = empty((1856, ), device='cpu', dtype=torch.float32)
    buf280 = empty((1856, ), device='cpu', dtype=torch.float32)
    buf281 = buf277; del buf277  # reuse
    buf282 = buf280; del buf280  # reuse
    buf283 = buf261; del buf261  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_38(c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(relu_73.data_ptr()), c_void_p(cat_45.data_ptr()), c_void_p(unsqueeze_890.data_ptr()), c_void_p(squeeze_220.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf283.data_ptr()))
    del cat_45
    del primals_147
    del relu_73
    del squeeze_220
    del unsqueeze_890
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf284 = aten.convolution_backward(buf283, relu_72, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_295
    buf285 = buf284[0]
    buf286 = buf284[1]
    del buf284
    buf287 = buf273; del buf273  # reuse
    buf288 = empty((800, ), device='cpu', dtype=torch.float32)
    buf289 = empty((800, ), device='cpu', dtype=torch.float32)
    buf290 = buf285; del buf285  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(buf290.data_ptr()), c_void_p(relu_72.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(unsqueeze_902.data_ptr()), c_void_p(squeeze_217.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    del convolution_71
    del primals_145
    del relu_72
    del squeeze_217
    del unsqueeze_902
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf291 = aten.convolution_backward(buf290, relu_71, primals_294, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf290
    del primals_294
    buf292 = buf291[0]
    buf293 = buf291[1]
    del buf291
    buf294 = buf288; del buf288  # reuse
    buf295 = empty((800, ), device='cpu', dtype=torch.float32)
    buf296 = empty((800, ), device='cpu', dtype=torch.float32)
    buf297 = buf292; del buf292  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40(c_void_p(buf297.data_ptr()), c_void_p(relu_71.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(unsqueeze_914.data_ptr()), c_void_p(squeeze_214.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    del convolution_70
    del primals_143
    del relu_71
    del squeeze_214
    del unsqueeze_914
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf298 = aten.convolution_backward(buf297, relu_70, primals_293, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf297
    del primals_293
    buf299 = buf298[0]
    buf300 = buf298[1]
    del buf298
    buf301 = empty((1792, ), device='cpu', dtype=torch.float32)
    buf302 = empty((1792, ), device='cpu', dtype=torch.float32)
    buf303 = buf299; del buf299  # reuse
    buf304 = buf302; del buf302  # reuse
    buf305 = buf283; del buf283  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_41(c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(relu_70.data_ptr()), c_void_p(cat_43.data_ptr()), c_void_p(unsqueeze_926.data_ptr()), c_void_p(squeeze_211.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf305.data_ptr()))
    del cat_43
    del primals_141
    del relu_70
    del squeeze_211
    del unsqueeze_926
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf306 = aten.convolution_backward(buf305, relu_69, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_292
    buf307 = buf306[0]
    buf308 = buf306[1]
    del buf306
    buf309 = buf295; del buf295  # reuse
    buf310 = empty((800, ), device='cpu', dtype=torch.float32)
    buf311 = empty((800, ), device='cpu', dtype=torch.float32)
    buf312 = buf307; del buf307  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42(c_void_p(buf312.data_ptr()), c_void_p(relu_69.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(unsqueeze_938.data_ptr()), c_void_p(squeeze_208.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()))
    del convolution_68
    del primals_139
    del relu_69
    del squeeze_208
    del unsqueeze_938
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf313 = aten.convolution_backward(buf312, relu_68, primals_291, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf312
    del primals_291
    buf314 = buf313[0]
    buf315 = buf313[1]
    del buf313
    buf316 = buf310; del buf310  # reuse
    buf317 = empty((800, ), device='cpu', dtype=torch.float32)
    buf318 = empty((800, ), device='cpu', dtype=torch.float32)
    buf319 = buf314; del buf314  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(buf319.data_ptr()), c_void_p(relu_68.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(unsqueeze_950.data_ptr()), c_void_p(squeeze_205.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()))
    del convolution_67
    del primals_137
    del relu_68
    del squeeze_205
    del unsqueeze_950
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf320 = aten.convolution_backward(buf319, relu_67, primals_290, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf319
    del primals_290
    buf321 = buf320[0]
    buf322 = buf320[1]
    del buf320
    buf323 = empty((1728, ), device='cpu', dtype=torch.float32)
    buf324 = empty((1728, ), device='cpu', dtype=torch.float32)
    buf325 = buf321; del buf321  # reuse
    buf326 = buf324; del buf324  # reuse
    buf327 = buf305; del buf305  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_44(c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(relu_67.data_ptr()), c_void_p(cat_41.data_ptr()), c_void_p(unsqueeze_962.data_ptr()), c_void_p(squeeze_202.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf327.data_ptr()))
    del cat_41
    del primals_135
    del relu_67
    del squeeze_202
    del unsqueeze_962
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf328 = aten.convolution_backward(buf327, relu_66, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_289
    buf329 = buf328[0]
    buf330 = buf328[1]
    del buf328
    buf331 = buf317; del buf317  # reuse
    buf332 = empty((800, ), device='cpu', dtype=torch.float32)
    buf333 = empty((800, ), device='cpu', dtype=torch.float32)
    buf334 = buf329; del buf329  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45(c_void_p(buf334.data_ptr()), c_void_p(relu_66.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_974.data_ptr()), c_void_p(squeeze_199.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    del convolution_65
    del primals_133
    del relu_66
    del squeeze_199
    del unsqueeze_974
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf335 = aten.convolution_backward(buf334, relu_65, primals_288, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf334
    del primals_288
    buf336 = buf335[0]
    buf337 = buf335[1]
    del buf335
    buf338 = buf332; del buf332  # reuse
    buf339 = empty((800, ), device='cpu', dtype=torch.float32)
    buf340 = empty((800, ), device='cpu', dtype=torch.float32)
    buf341 = buf336; del buf336  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_46(c_void_p(buf341.data_ptr()), c_void_p(relu_65.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_986.data_ptr()), c_void_p(squeeze_196.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    del convolution_64
    del primals_131
    del relu_65
    del squeeze_196
    del unsqueeze_986
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf342 = aten.convolution_backward(buf341, relu_64, primals_287, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf341
    del primals_287
    buf343 = buf342[0]
    buf344 = buf342[1]
    del buf342
    buf345 = empty((1664, ), device='cpu', dtype=torch.float32)
    buf346 = empty((1664, ), device='cpu', dtype=torch.float32)
    buf347 = buf343; del buf343  # reuse
    buf348 = buf346; del buf346  # reuse
    buf349 = buf259; del buf259  # reuse
    buf350 = reinterpret_tensor(buf28, (8, 640, 14, 14), (125440, 1, 8960, 640), 0); del buf28  # reuse
    buf351 = buf327; del buf327  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_47(c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(relu_64.data_ptr()), c_void_p(cat_39.data_ptr()), c_void_p(unsqueeze_998.data_ptr()), c_void_p(squeeze_193.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    del buf260
    del buf281
    del buf303
    del buf325
    del buf347
    del cat_39
    del primals_129
    del relu_64
    del squeeze_193
    del unsqueeze_998
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf352 = aten.convolution_backward(buf351, relu_63, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_286
    buf353 = buf352[0]
    buf354 = buf352[1]
    del buf352
    buf355 = buf339; del buf339  # reuse
    buf356 = empty((800, ), device='cpu', dtype=torch.float32)
    buf357 = empty((800, ), device='cpu', dtype=torch.float32)
    buf358 = buf353; del buf353  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48(c_void_p(buf358.data_ptr()), c_void_p(relu_63.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(unsqueeze_1010.data_ptr()), c_void_p(squeeze_190.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()))
    del convolution_62
    del primals_127
    del relu_63
    del squeeze_190
    del unsqueeze_1010
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf359 = aten.convolution_backward(buf358, relu_62, primals_285, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf358
    del primals_285
    buf360 = buf359[0]
    buf361 = buf359[1]
    del buf359
    buf362 = buf356; del buf356  # reuse
    buf363 = empty((800, ), device='cpu', dtype=torch.float32)
    buf364 = empty((800, ), device='cpu', dtype=torch.float32)
    buf365 = buf360; del buf360  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49(c_void_p(buf365.data_ptr()), c_void_p(relu_62.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_1022.data_ptr()), c_void_p(squeeze_187.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()))
    del convolution_61
    del primals_125
    del relu_62
    del squeeze_187
    del unsqueeze_1022
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf366 = aten.convolution_backward(buf365, relu_61, primals_284, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf365
    del primals_284
    buf367 = buf366[0]
    buf368 = buf366[1]
    del buf366
    buf369 = buf64; del buf64  # reuse
    buf370 = empty((1600, ), device='cpu', dtype=torch.float32)
    buf371 = buf367; del buf367  # reuse
    buf372 = buf370; del buf370  # reuse
    buf373 = buf351; del buf351  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_50(c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(relu_61.data_ptr()), c_void_p(cat_37.data_ptr()), c_void_p(unsqueeze_1034.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf373.data_ptr()))
    del cat_37
    del primals_123
    del relu_61
    del squeeze_184
    del unsqueeze_1034
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf374 = aten.convolution_backward(buf373, relu_60, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_283
    buf375 = buf374[0]
    buf376 = buf374[1]
    del buf374
    buf377 = buf363; del buf363  # reuse
    buf378 = empty((800, ), device='cpu', dtype=torch.float32)
    buf379 = empty((800, ), device='cpu', dtype=torch.float32)
    buf380 = buf375; del buf375  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51(c_void_p(buf380.data_ptr()), c_void_p(relu_60.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_1046.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    del convolution_59
    del primals_121
    del relu_60
    del squeeze_181
    del unsqueeze_1046
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf381 = aten.convolution_backward(buf380, relu_59, primals_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf380
    del primals_282
    buf382 = buf381[0]
    buf383 = buf381[1]
    del buf381
    buf384 = buf378; del buf378  # reuse
    buf385 = empty((800, ), device='cpu', dtype=torch.float32)
    buf386 = empty((800, ), device='cpu', dtype=torch.float32)
    buf387 = buf382; del buf382  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52(c_void_p(buf387.data_ptr()), c_void_p(relu_59.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(unsqueeze_1058.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    del convolution_58
    del primals_119
    del relu_59
    del squeeze_178
    del unsqueeze_1058
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf388 = aten.convolution_backward(buf387, relu_58, primals_281, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf387
    del primals_281
    buf389 = buf388[0]
    buf390 = buf388[1]
    del buf388
    buf391 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf392 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf393 = buf389; del buf389  # reuse
    buf394 = buf392; del buf392  # reuse
    buf395 = buf373; del buf373  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_53(c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(relu_58.data_ptr()), c_void_p(cat_35.data_ptr()), c_void_p(unsqueeze_1070.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf395.data_ptr()))
    del cat_35
    del primals_117
    del relu_58
    del squeeze_175
    del unsqueeze_1070
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf396 = aten.convolution_backward(buf395, relu_57, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_280
    buf397 = buf396[0]
    buf398 = buf396[1]
    del buf396
    buf399 = buf385; del buf385  # reuse
    buf400 = empty((800, ), device='cpu', dtype=torch.float32)
    buf401 = empty((800, ), device='cpu', dtype=torch.float32)
    buf402 = buf397; del buf397  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54(c_void_p(buf402.data_ptr()), c_void_p(relu_57.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_1082.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()))
    del convolution_56
    del primals_115
    del relu_57
    del squeeze_172
    del unsqueeze_1082
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf403 = aten.convolution_backward(buf402, relu_56, primals_279, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf402
    del primals_279
    buf404 = buf403[0]
    buf405 = buf403[1]
    del buf403
    buf406 = buf400; del buf400  # reuse
    buf407 = empty((800, ), device='cpu', dtype=torch.float32)
    buf408 = empty((800, ), device='cpu', dtype=torch.float32)
    buf409 = buf404; del buf404  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55(c_void_p(buf409.data_ptr()), c_void_p(relu_56.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_1094.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()))
    del convolution_55
    del primals_113
    del relu_56
    del squeeze_169
    del unsqueeze_1094
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf410 = aten.convolution_backward(buf409, relu_55, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf409
    del primals_278
    buf411 = buf410[0]
    buf412 = buf410[1]
    del buf410
    buf413 = empty((1472, ), device='cpu', dtype=torch.float32)
    buf414 = empty((1472, ), device='cpu', dtype=torch.float32)
    buf415 = buf411; del buf411  # reuse
    buf416 = buf414; del buf414  # reuse
    buf417 = buf395; del buf395  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_56(c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(relu_55.data_ptr()), c_void_p(cat_33.data_ptr()), c_void_p(unsqueeze_1106.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf417.data_ptr()))
    del cat_33
    del primals_111
    del relu_55
    del squeeze_166
    del unsqueeze_1106
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf418 = aten.convolution_backward(buf417, relu_54, primals_277, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_277
    buf419 = buf418[0]
    buf420 = buf418[1]
    del buf418
    buf421 = buf407; del buf407  # reuse
    buf422 = empty((800, ), device='cpu', dtype=torch.float32)
    buf423 = empty((800, ), device='cpu', dtype=torch.float32)
    buf424 = buf419; del buf419  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_57(c_void_p(buf424.data_ptr()), c_void_p(relu_54.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_1118.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()))
    del convolution_53
    del primals_109
    del relu_54
    del squeeze_163
    del unsqueeze_1118
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf425 = aten.convolution_backward(buf424, relu_53, primals_276, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf424
    del primals_276
    buf426 = buf425[0]
    buf427 = buf425[1]
    del buf425
    buf428 = buf422; del buf422  # reuse
    buf429 = empty((800, ), device='cpu', dtype=torch.float32)
    buf430 = empty((800, ), device='cpu', dtype=torch.float32)
    buf431 = buf426; del buf426  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58(c_void_p(buf431.data_ptr()), c_void_p(relu_53.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_1130.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()))
    del convolution_52
    del primals_107
    del relu_53
    del squeeze_160
    del unsqueeze_1130
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf432 = aten.convolution_backward(buf431, relu_52, primals_275, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf431
    del primals_275
    buf433 = buf432[0]
    buf434 = buf432[1]
    del buf432
    buf435 = empty((1408, ), device='cpu', dtype=torch.float32)
    buf436 = empty((1408, ), device='cpu', dtype=torch.float32)
    buf437 = buf433; del buf433  # reuse
    buf438 = buf436; del buf436  # reuse
    buf439 = buf349; del buf349  # reuse
    buf440 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf441 = buf417; del buf417  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_59(c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(relu_52.data_ptr()), c_void_p(cat_31.data_ptr()), c_void_p(unsqueeze_1142.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()))
    del buf350
    del buf371
    del buf415
    del buf437
    del cat_31
    del primals_105
    del relu_52
    del squeeze_157
    del unsqueeze_1142
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf442 = aten.convolution_backward(buf441, relu_51, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_274
    buf443 = buf442[0]
    buf444 = buf442[1]
    del buf442
    buf445 = buf429; del buf429  # reuse
    buf446 = empty((800, ), device='cpu', dtype=torch.float32)
    buf447 = empty((800, ), device='cpu', dtype=torch.float32)
    buf448 = buf443; del buf443  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60(c_void_p(buf448.data_ptr()), c_void_p(relu_51.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_1154.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()))
    del convolution_50
    del primals_103
    del relu_51
    del squeeze_154
    del unsqueeze_1154
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf449 = aten.convolution_backward(buf448, relu_50, primals_273, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf448
    del primals_273
    buf450 = buf449[0]
    buf451 = buf449[1]
    del buf449
    buf452 = buf446; del buf446  # reuse
    buf453 = empty((800, ), device='cpu', dtype=torch.float32)
    buf454 = empty((800, ), device='cpu', dtype=torch.float32)
    buf455 = buf450; del buf450  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61(c_void_p(buf455.data_ptr()), c_void_p(relu_50.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_1166.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()))
    del convolution_49
    del primals_101
    del relu_50
    del squeeze_151
    del unsqueeze_1166
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf456 = aten.convolution_backward(buf455, relu_49, primals_272, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf455
    del primals_272
    buf457 = buf456[0]
    buf458 = buf456[1]
    del buf456
    buf459 = empty((1344, ), device='cpu', dtype=torch.float32)
    buf460 = empty((1344, ), device='cpu', dtype=torch.float32)
    buf461 = buf457; del buf457  # reuse
    buf462 = buf460; del buf460  # reuse
    buf463 = buf441; del buf441  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_62(c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(relu_49.data_ptr()), c_void_p(cat_29.data_ptr()), c_void_p(unsqueeze_1178.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf463.data_ptr()))
    del cat_29
    del primals_99
    del relu_49
    del squeeze_148
    del unsqueeze_1178
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf464 = aten.convolution_backward(buf463, relu_48, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_271
    buf465 = buf464[0]
    buf466 = buf464[1]
    del buf464
    buf467 = buf453; del buf453  # reuse
    buf468 = empty((800, ), device='cpu', dtype=torch.float32)
    buf469 = empty((800, ), device='cpu', dtype=torch.float32)
    buf470 = buf465; del buf465  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_63(c_void_p(buf470.data_ptr()), c_void_p(relu_48.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_1190.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()))
    del convolution_47
    del primals_97
    del relu_48
    del squeeze_145
    del unsqueeze_1190
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf471 = aten.convolution_backward(buf470, relu_47, primals_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf470
    del primals_270
    buf472 = buf471[0]
    buf473 = buf471[1]
    del buf471
    buf474 = buf468; del buf468  # reuse
    buf475 = empty((800, ), device='cpu', dtype=torch.float32)
    buf476 = empty((800, ), device='cpu', dtype=torch.float32)
    buf477 = buf472; del buf472  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64(c_void_p(buf477.data_ptr()), c_void_p(relu_47.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_1202.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()))
    del convolution_46
    del primals_95
    del relu_47
    del squeeze_142
    del unsqueeze_1202
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf478 = aten.convolution_backward(buf477, relu_46, primals_269, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf477
    del primals_269
    buf479 = buf478[0]
    buf480 = buf478[1]
    del buf478
    buf481 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf482 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf483 = buf479; del buf479  # reuse
    buf484 = buf482; del buf482  # reuse
    buf485 = buf463; del buf463  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_65(c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(cat_27.data_ptr()), c_void_p(unsqueeze_1214.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf485.data_ptr()))
    del cat_27
    del primals_93
    del relu_46
    del squeeze_139
    del unsqueeze_1214
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf486 = aten.convolution_backward(buf485, relu_45, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_268
    buf487 = buf486[0]
    buf488 = buf486[1]
    del buf486
    buf489 = buf475; del buf475  # reuse
    buf490 = empty((800, ), device='cpu', dtype=torch.float32)
    buf491 = empty((800, ), device='cpu', dtype=torch.float32)
    buf492 = buf487; del buf487  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66(c_void_p(buf492.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_1226.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()))
    del convolution_44
    del primals_91
    del relu_45
    del squeeze_136
    del unsqueeze_1226
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf493 = aten.convolution_backward(buf492, relu_44, primals_267, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf492
    del primals_267
    buf494 = buf493[0]
    buf495 = buf493[1]
    del buf493
    buf496 = buf490; del buf490  # reuse
    buf497 = empty((800, ), device='cpu', dtype=torch.float32)
    buf498 = empty((800, ), device='cpu', dtype=torch.float32)
    buf499 = buf494; del buf494  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67(c_void_p(buf499.data_ptr()), c_void_p(relu_44.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_1238.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()))
    del convolution_43
    del primals_89
    del relu_44
    del squeeze_133
    del unsqueeze_1238
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf500 = aten.convolution_backward(buf499, relu_43, primals_266, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf499
    del primals_266
    buf501 = buf500[0]
    buf502 = buf500[1]
    del buf500
    buf503 = empty((1216, ), device='cpu', dtype=torch.float32)
    buf504 = empty((1216, ), device='cpu', dtype=torch.float32)
    buf505 = buf501; del buf501  # reuse
    buf506 = buf504; del buf504  # reuse
    buf507 = buf485; del buf485  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_68(c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(relu_43.data_ptr()), c_void_p(cat_25.data_ptr()), c_void_p(unsqueeze_1250.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf507.data_ptr()))
    del cat_25
    del primals_87
    del relu_43
    del squeeze_130
    del unsqueeze_1250
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf508 = aten.convolution_backward(buf507, relu_42, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf507
    del primals_265
    buf509 = buf508[0]
    buf510 = buf508[1]
    del buf508
    buf511 = buf497; del buf497  # reuse
    buf512 = empty((800, ), device='cpu', dtype=torch.float32)
    buf513 = empty((800, ), device='cpu', dtype=torch.float32)
    buf514 = buf509; del buf509  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69(c_void_p(buf514.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_1262.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()))
    del convolution_41
    del primals_85
    del relu_42
    del squeeze_127
    del unsqueeze_1262
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf515 = aten.convolution_backward(buf514, relu_41, primals_264, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf514
    del primals_264
    buf516 = buf515[0]
    buf517 = buf515[1]
    del buf515
    buf518 = buf512; del buf512  # reuse
    buf519 = empty((800, ), device='cpu', dtype=torch.float32)
    buf520 = empty((800, ), device='cpu', dtype=torch.float32)
    buf521 = buf516; del buf516  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70(c_void_p(buf521.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_1274.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf520.data_ptr()))
    del buf519
    del convolution_40
    del primals_83
    del relu_41
    del squeeze_124
    del unsqueeze_1274
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf522 = aten.convolution_backward(buf521, relu_40, primals_263, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf521
    del primals_263
    buf523 = buf522[0]
    buf524 = buf522[1]
    del buf522
    buf528 = reinterpret_tensor(buf170, (8, 1152, 14, 14), (225792, 196, 14, 1), 0); del buf170  # reuse
    cpp_fused_add_convolution_backward_slice_backward_71(c_void_p(buf440.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf528.data_ptr()))
    del buf439
    del buf440
    del buf461
    del buf483
    del buf505
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf529 = aten.convolution_backward(buf528, relu_39, primals_262, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf528
    del primals_262
    buf530 = buf529[0]
    buf525 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf526 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf532 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf533 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf527 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf534 = empty((1152, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_threshold_backward_72(c_void_p(relu_40.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(cat_23.data_ptr()), c_void_p(unsqueeze_1286.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf534.data_ptr()))
    buf531 = buf529[1]
    del buf529
    buf535 = buf523; del buf523  # reuse
    buf536 = reinterpret_tensor(buf123, (8, 576, 28, 28), (451584, 784, 28, 1), 0); del buf123  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_73(c_void_p(buf535.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(cat_23.data_ptr()), c_void_p(unsqueeze_1286.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf536.data_ptr()))
    del buf526
    del buf530
    del buf533
    del cat_23
    del primals_79
    del primals_81
    del relu_39
    del relu_40
    del squeeze_118
    del unsqueeze_1286
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf537 = aten.convolution_backward(buf536, relu_38, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_261
    buf538 = buf537[0]
    buf539 = buf537[1]
    del buf537
    buf540 = empty((400, ), device='cpu', dtype=torch.float32)
    buf541 = empty((400, ), device='cpu', dtype=torch.float32)
    buf542 = empty((400, ), device='cpu', dtype=torch.float32)
    buf543 = buf538; del buf538  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_74(c_void_p(buf543.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_1310.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()))
    del convolution_37
    del primals_77
    del relu_38
    del squeeze_115
    del unsqueeze_1310
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf544 = aten.convolution_backward(buf543, relu_37, primals_260, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf543
    del primals_260
    buf545 = buf544[0]
    buf546 = buf544[1]
    del buf544
    buf547 = buf541; del buf541  # reuse
    buf548 = empty((400, ), device='cpu', dtype=torch.float32)
    buf549 = empty((400, ), device='cpu', dtype=torch.float32)
    buf550 = buf545; del buf545  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75(c_void_p(buf550.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_1322.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()))
    del convolution_36
    del primals_75
    del relu_37
    del squeeze_112
    del unsqueeze_1322
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf551 = aten.convolution_backward(buf550, relu_36, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf550
    del primals_259
    buf552 = buf551[0]
    buf553 = buf551[1]
    del buf551
    buf554 = empty((1088, ), device='cpu', dtype=torch.float32)
    buf555 = empty((1088, ), device='cpu', dtype=torch.float32)
    buf556 = buf552; del buf552  # reuse
    buf557 = buf555; del buf555  # reuse
    buf558 = buf536; del buf536  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_76(c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(cat_21.data_ptr()), c_void_p(unsqueeze_1334.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf558.data_ptr()))
    del cat_21
    del primals_73
    del relu_36
    del squeeze_109
    del unsqueeze_1334
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf559 = aten.convolution_backward(buf558, relu_35, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_258
    buf560 = buf559[0]
    buf561 = buf559[1]
    del buf559
    buf562 = buf548; del buf548  # reuse
    buf563 = empty((400, ), device='cpu', dtype=torch.float32)
    buf564 = empty((400, ), device='cpu', dtype=torch.float32)
    buf565 = buf560; del buf560  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77(c_void_p(buf565.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_1346.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()))
    del convolution_34
    del primals_71
    del relu_35
    del squeeze_106
    del unsqueeze_1346
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf566 = aten.convolution_backward(buf565, relu_34, primals_257, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf565
    del primals_257
    buf567 = buf566[0]
    buf568 = buf566[1]
    del buf566
    buf569 = buf563; del buf563  # reuse
    buf570 = empty((400, ), device='cpu', dtype=torch.float32)
    buf571 = empty((400, ), device='cpu', dtype=torch.float32)
    buf572 = buf567; del buf567  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78(c_void_p(buf572.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_1358.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()))
    del convolution_33
    del primals_69
    del relu_34
    del squeeze_103
    del unsqueeze_1358
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf573 = aten.convolution_backward(buf572, relu_33, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf572
    del primals_256
    buf574 = buf573[0]
    buf575 = buf573[1]
    del buf573
    buf576 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf577 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf578 = buf574; del buf574  # reuse
    buf579 = buf577; del buf577  # reuse
    buf580 = buf558; del buf558  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_79(c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(cat_19.data_ptr()), c_void_p(unsqueeze_1370.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf580.data_ptr()))
    del cat_19
    del primals_67
    del relu_33
    del squeeze_100
    del unsqueeze_1370
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf581 = aten.convolution_backward(buf580, relu_32, primals_255, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_255
    buf582 = buf581[0]
    buf583 = buf581[1]
    del buf581
    buf584 = buf570; del buf570  # reuse
    buf585 = empty((400, ), device='cpu', dtype=torch.float32)
    buf586 = empty((400, ), device='cpu', dtype=torch.float32)
    buf587 = buf582; del buf582  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_80(c_void_p(buf587.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_1382.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()))
    del convolution_31
    del primals_65
    del relu_32
    del squeeze_97
    del unsqueeze_1382
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf588 = aten.convolution_backward(buf587, relu_31, primals_254, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf587
    del primals_254
    buf589 = buf588[0]
    buf590 = buf588[1]
    del buf588
    buf591 = buf585; del buf585  # reuse
    buf592 = empty((400, ), device='cpu', dtype=torch.float32)
    buf593 = empty((400, ), device='cpu', dtype=torch.float32)
    buf594 = buf589; del buf589  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81(c_void_p(buf594.data_ptr()), c_void_p(relu_31.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_1394.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()))
    del convolution_30
    del primals_63
    del relu_31
    del squeeze_94
    del unsqueeze_1394
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf595 = aten.convolution_backward(buf594, relu_30, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf594
    del primals_253
    buf596 = buf595[0]
    buf597 = buf595[1]
    del buf595
    buf598 = empty((960, ), device='cpu', dtype=torch.float32)
    buf599 = empty((960, ), device='cpu', dtype=torch.float32)
    buf600 = buf596; del buf596  # reuse
    buf601 = buf599; del buf599  # reuse
    buf602 = buf580; del buf580  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_82(c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(cat_17.data_ptr()), c_void_p(unsqueeze_1406.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf602.data_ptr()))
    del cat_17
    del primals_61
    del relu_30
    del squeeze_91
    del unsqueeze_1406
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf603 = aten.convolution_backward(buf602, relu_29, primals_252, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_252
    buf604 = buf603[0]
    buf605 = buf603[1]
    del buf603
    buf606 = buf592; del buf592  # reuse
    buf607 = empty((400, ), device='cpu', dtype=torch.float32)
    buf608 = empty((400, ), device='cpu', dtype=torch.float32)
    buf609 = buf604; del buf604  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83(c_void_p(buf609.data_ptr()), c_void_p(relu_29.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_1418.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf608.data_ptr()))
    del convolution_28
    del primals_59
    del relu_29
    del squeeze_88
    del unsqueeze_1418
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf610 = aten.convolution_backward(buf609, relu_28, primals_251, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf609
    del primals_251
    buf611 = buf610[0]
    buf612 = buf610[1]
    del buf610
    buf613 = buf607; del buf607  # reuse
    buf614 = empty((400, ), device='cpu', dtype=torch.float32)
    buf615 = empty((400, ), device='cpu', dtype=torch.float32)
    buf616 = buf611; del buf611  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84(c_void_p(buf616.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_1430.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()))
    del convolution_27
    del primals_57
    del relu_28
    del squeeze_85
    del unsqueeze_1430
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf617 = aten.convolution_backward(buf616, relu_27, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf616
    del primals_250
    buf618 = buf617[0]
    buf619 = buf617[1]
    del buf617
    buf620 = empty((896, ), device='cpu', dtype=torch.float32)
    buf621 = empty((896, ), device='cpu', dtype=torch.float32)
    buf622 = buf618; del buf618  # reuse
    buf623 = buf621; del buf621  # reuse
    buf624 = reinterpret_tensor(buf213, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf213  # reuse
    buf625 = reinterpret_tensor(buf393, (8, 384, 28, 28), (301056, 1, 10752, 384), 0); del buf393  # reuse
    buf626 = buf602; del buf602  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_85(c_void_p(buf622.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(cat_15.data_ptr()), c_void_p(unsqueeze_1442.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf626.data_ptr()))
    del buf535
    del buf556
    del buf578
    del buf600
    del buf622
    del cat_15
    del primals_55
    del relu_27
    del squeeze_82
    del unsqueeze_1442
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf627 = aten.convolution_backward(buf626, relu_26, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_249
    buf628 = buf627[0]
    buf629 = buf627[1]
    del buf627
    buf630 = buf614; del buf614  # reuse
    buf631 = empty((400, ), device='cpu', dtype=torch.float32)
    buf632 = empty((400, ), device='cpu', dtype=torch.float32)
    buf633 = buf628; del buf628  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_86(c_void_p(buf633.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_1454.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()))
    del convolution_25
    del primals_53
    del relu_26
    del squeeze_79
    del unsqueeze_1454
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf634 = aten.convolution_backward(buf633, relu_25, primals_248, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf633
    del primals_248
    buf635 = buf634[0]
    buf636 = buf634[1]
    del buf634
    buf637 = buf631; del buf631  # reuse
    buf638 = empty((400, ), device='cpu', dtype=torch.float32)
    buf639 = empty((400, ), device='cpu', dtype=torch.float32)
    buf640 = buf635; del buf635  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_87(c_void_p(buf640.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_1466.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()))
    del convolution_24
    del primals_51
    del relu_25
    del squeeze_76
    del unsqueeze_1466
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf641 = aten.convolution_backward(buf640, relu_24, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf640
    del primals_247
    buf642 = buf641[0]
    buf643 = buf641[1]
    del buf641
    buf644 = empty((832, ), device='cpu', dtype=torch.float32)
    buf645 = empty((832, ), device='cpu', dtype=torch.float32)
    buf646 = buf642; del buf642  # reuse
    buf647 = buf645; del buf645  # reuse
    buf648 = buf626; del buf626  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_88(c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(cat_13.data_ptr()), c_void_p(unsqueeze_1478.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf648.data_ptr()))
    del cat_13
    del primals_49
    del relu_24
    del squeeze_73
    del unsqueeze_1478
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf649 = aten.convolution_backward(buf648, relu_23, primals_246, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_246
    buf650 = buf649[0]
    buf651 = buf649[1]
    del buf649
    buf652 = buf638; del buf638  # reuse
    buf653 = empty((400, ), device='cpu', dtype=torch.float32)
    buf654 = empty((400, ), device='cpu', dtype=torch.float32)
    buf655 = buf650; del buf650  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_89(c_void_p(buf655.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_1490.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf654.data_ptr()))
    del convolution_22
    del primals_47
    del relu_23
    del squeeze_70
    del unsqueeze_1490
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf656 = aten.convolution_backward(buf655, relu_22, primals_245, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf655
    del primals_245
    buf657 = buf656[0]
    buf658 = buf656[1]
    del buf656
    buf659 = buf653; del buf653  # reuse
    buf660 = empty((400, ), device='cpu', dtype=torch.float32)
    buf661 = empty((400, ), device='cpu', dtype=torch.float32)
    buf662 = buf657; del buf657  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90(c_void_p(buf662.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_1502.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()))
    del convolution_21
    del primals_45
    del relu_22
    del squeeze_67
    del unsqueeze_1502
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf663 = aten.convolution_backward(buf662, relu_21, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf662
    del primals_244
    buf664 = buf663[0]
    buf665 = buf663[1]
    del buf663
    buf666 = empty((768, ), device='cpu', dtype=torch.float32)
    buf667 = empty((768, ), device='cpu', dtype=torch.float32)
    buf668 = buf664; del buf664  # reuse
    buf669 = buf667; del buf667  # reuse
    buf670 = buf648; del buf648  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_91(c_void_p(buf668.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(cat_11.data_ptr()), c_void_p(unsqueeze_1514.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf670.data_ptr()))
    del cat_11
    del primals_43
    del relu_21
    del squeeze_64
    del unsqueeze_1514
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf671 = aten.convolution_backward(buf670, relu_20, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_243
    buf672 = buf671[0]
    buf673 = buf671[1]
    del buf671
    buf674 = buf660; del buf660  # reuse
    buf675 = empty((400, ), device='cpu', dtype=torch.float32)
    buf676 = empty((400, ), device='cpu', dtype=torch.float32)
    buf677 = buf672; del buf672  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92(c_void_p(buf677.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_1526.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf676.data_ptr()))
    del convolution_19
    del primals_41
    del relu_20
    del squeeze_61
    del unsqueeze_1526
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf678 = aten.convolution_backward(buf677, relu_19, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf677
    del primals_242
    buf679 = buf678[0]
    buf680 = buf678[1]
    del buf678
    buf681 = buf675; del buf675  # reuse
    buf682 = empty((400, ), device='cpu', dtype=torch.float32)
    buf683 = empty((400, ), device='cpu', dtype=torch.float32)
    buf684 = buf679; del buf679  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93(c_void_p(buf684.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_1538.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf683.data_ptr()))
    del convolution_18
    del primals_39
    del relu_19
    del squeeze_58
    del unsqueeze_1538
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf685 = aten.convolution_backward(buf684, relu_18, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf684
    del primals_241
    buf686 = buf685[0]
    buf687 = buf685[1]
    del buf685
    buf688 = empty((704, ), device='cpu', dtype=torch.float32)
    buf689 = empty((704, ), device='cpu', dtype=torch.float32)
    buf690 = buf686; del buf686  # reuse
    buf691 = buf689; del buf689  # reuse
    buf692 = buf670; del buf670  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_94(c_void_p(buf690.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(cat_9.data_ptr()), c_void_p(unsqueeze_1550.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf692.data_ptr()))
    del cat_9
    del primals_37
    del relu_18
    del squeeze_55
    del unsqueeze_1550
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf693 = aten.convolution_backward(buf692, relu_17, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf692
    del primals_240
    buf694 = buf693[0]
    buf695 = buf693[1]
    del buf693
    buf696 = buf682; del buf682  # reuse
    buf697 = empty((400, ), device='cpu', dtype=torch.float32)
    buf698 = empty((400, ), device='cpu', dtype=torch.float32)
    buf699 = buf694; del buf694  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_95(c_void_p(buf699.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_1562.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf698.data_ptr()))
    del convolution_16
    del primals_35
    del relu_17
    del squeeze_52
    del unsqueeze_1562
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf700 = aten.convolution_backward(buf699, relu_16, primals_239, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf699
    del primals_239
    buf701 = buf700[0]
    buf702 = buf700[1]
    del buf700
    buf703 = buf697; del buf697  # reuse
    buf704 = empty((400, ), device='cpu', dtype=torch.float32)
    buf705 = empty((400, ), device='cpu', dtype=torch.float32)
    buf706 = buf701; del buf701  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_96(c_void_p(buf706.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_1574.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf703.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf705.data_ptr()))
    del buf704
    del convolution_15
    del primals_33
    del relu_16
    del squeeze_49
    del unsqueeze_1574
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf707 = aten.convolution_backward(buf706, relu_15, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf706
    del primals_238
    buf708 = buf707[0]
    buf709 = buf707[1]
    del buf707
    buf713 = empty((8, 640, 28, 28), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_slice_backward_97(c_void_p(buf625.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf713.data_ptr()))
    del buf624
    del buf625
    del buf646
    del buf668
    del buf690
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf714 = aten.convolution_backward(buf713, relu_14, primals_237, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf713
    del primals_237
    buf715 = buf714[0]
    buf710 = empty((376, ), device='cpu', dtype=torch.float32)
    buf711 = empty((376, ), device='cpu', dtype=torch.float32)
    buf717 = empty((376, ), device='cpu', dtype=torch.float32)
    buf718 = empty((376, ), device='cpu', dtype=torch.float32)
    buf712 = empty((376, ), device='cpu', dtype=torch.float32)
    buf719 = empty((376, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_threshold_backward_98(c_void_p(relu_15.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(cat_7.data_ptr()), c_void_p(unsqueeze_1586.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf719.data_ptr()))
    buf716 = buf714[1]
    del buf714
    buf720 = buf708; del buf708  # reuse
    buf721 = empty((8, 276, 56, 56), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_99(c_void_p(buf720.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(cat_7.data_ptr()), c_void_p(unsqueeze_1586.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf721.data_ptr()))
    del buf711
    del buf715
    del buf718
    del cat_7
    del primals_29
    del primals_31
    del relu_14
    del relu_15
    del squeeze_43
    del unsqueeze_1586
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf722 = aten.convolution_backward(buf721, relu_13, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_236
    buf723 = buf722[0]
    buf724 = buf722[1]
    del buf722
    buf725 = empty((200, ), device='cpu', dtype=torch.float32)
    buf726 = empty((200, ), device='cpu', dtype=torch.float32)
    buf727 = empty((200, ), device='cpu', dtype=torch.float32)
    buf728 = buf723; del buf723  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_100(c_void_p(buf728.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_1610.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf727.data_ptr()))
    del convolution_12
    del primals_27
    del relu_13
    del squeeze_40
    del unsqueeze_1610
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf729 = aten.convolution_backward(buf728, relu_12, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf728
    del primals_235
    buf730 = buf729[0]
    buf731 = buf729[1]
    del buf729
    buf732 = buf726; del buf726  # reuse
    buf733 = empty((200, ), device='cpu', dtype=torch.float32)
    buf734 = empty((200, ), device='cpu', dtype=torch.float32)
    buf735 = buf730; del buf730  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_101(c_void_p(buf735.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_1622.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf732.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf734.data_ptr()))
    del convolution_11
    del primals_25
    del relu_12
    del squeeze_37
    del unsqueeze_1622
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf736 = aten.convolution_backward(buf735, relu_11, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf735
    del primals_234
    buf737 = buf736[0]
    buf738 = buf736[1]
    del buf736
    buf739 = empty((356, ), device='cpu', dtype=torch.float32)
    buf740 = empty((356, ), device='cpu', dtype=torch.float32)
    buf741 = buf737; del buf737  # reuse
    buf742 = buf740; del buf740  # reuse
    buf743 = buf721; del buf721  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_102(c_void_p(buf741.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(cat_5.data_ptr()), c_void_p(unsqueeze_1634.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf743.data_ptr()))
    del cat_5
    del primals_23
    del relu_11
    del squeeze_34
    del unsqueeze_1634
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf744 = aten.convolution_backward(buf743, relu_10, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_233
    buf745 = buf744[0]
    buf746 = buf744[1]
    del buf744
    buf747 = buf733; del buf733  # reuse
    buf748 = empty((200, ), device='cpu', dtype=torch.float32)
    buf749 = empty((200, ), device='cpu', dtype=torch.float32)
    buf750 = buf745; del buf745  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_103(c_void_p(buf750.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_1646.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf748.data_ptr()), c_void_p(buf749.data_ptr()))
    del convolution_9
    del primals_21
    del relu_10
    del squeeze_31
    del unsqueeze_1646
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf751 = aten.convolution_backward(buf750, relu_9, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf750
    del primals_232
    buf752 = buf751[0]
    buf753 = buf751[1]
    del buf751
    buf754 = buf748; del buf748  # reuse
    buf755 = empty((200, ), device='cpu', dtype=torch.float32)
    buf756 = empty((200, ), device='cpu', dtype=torch.float32)
    buf757 = buf752; del buf752  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_104(c_void_p(buf757.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_1658.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf756.data_ptr()))
    del convolution_8
    del primals_19
    del relu_9
    del squeeze_28
    del unsqueeze_1658
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf758 = aten.convolution_backward(buf757, relu_8, primals_231, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf757
    del primals_231
    buf759 = buf758[0]
    buf760 = buf758[1]
    del buf758
    buf761 = empty((336, ), device='cpu', dtype=torch.float32)
    buf762 = empty((336, ), device='cpu', dtype=torch.float32)
    buf763 = buf759; del buf759  # reuse
    buf764 = buf762; del buf762  # reuse
    buf765 = buf743; del buf743  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_105(c_void_p(buf763.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(cat_3.data_ptr()), c_void_p(unsqueeze_1670.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf765.data_ptr()))
    del cat_3
    del primals_17
    del relu_8
    del squeeze_25
    del unsqueeze_1670
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf766 = aten.convolution_backward(buf765, relu_7, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_230
    buf767 = buf766[0]
    buf768 = buf766[1]
    del buf766
    buf769 = buf755; del buf755  # reuse
    buf770 = empty((200, ), device='cpu', dtype=torch.float32)
    buf771 = empty((200, ), device='cpu', dtype=torch.float32)
    buf772 = buf767; del buf767  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_106(c_void_p(buf772.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_1682.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf771.data_ptr()))
    del convolution_6
    del primals_15
    del relu_7
    del squeeze_22
    del unsqueeze_1682
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf773 = aten.convolution_backward(buf772, relu_6, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf772
    del primals_229
    buf774 = buf773[0]
    buf775 = buf773[1]
    del buf773
    buf776 = buf770; del buf770  # reuse
    buf777 = empty((200, ), device='cpu', dtype=torch.float32)
    buf778 = empty((200, ), device='cpu', dtype=torch.float32)
    buf779 = buf774; del buf774  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_107(c_void_p(buf779.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_1694.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(buf778.data_ptr()))
    del convolution_5
    del primals_13
    del relu_6
    del squeeze_19
    del unsqueeze_1694
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf780 = aten.convolution_backward(buf779, relu_5, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf779
    del primals_228
    buf781 = buf780[0]
    buf782 = buf780[1]
    del buf780
    buf783 = empty((316, ), device='cpu', dtype=torch.float32)
    buf784 = empty((316, ), device='cpu', dtype=torch.float32)
    buf785 = buf781; del buf781  # reuse
    buf786 = buf784; del buf784  # reuse
    buf787 = buf765; del buf765  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_threshold_backward_108(c_void_p(buf785.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(cat_1.data_ptr()), c_void_p(unsqueeze_1706.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf787.data_ptr()))
    del cat_1
    del primals_11
    del relu_5
    del squeeze_16
    del unsqueeze_1706
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf788 = aten.convolution_backward(buf787, relu_4, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf787
    del primals_227
    buf789 = buf788[0]
    buf790 = buf788[1]
    del buf788
    buf791 = buf777; del buf777  # reuse
    buf792 = empty((200, ), device='cpu', dtype=torch.float32)
    buf793 = empty((200, ), device='cpu', dtype=torch.float32)
    buf794 = buf789; del buf789  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_109(c_void_p(buf794.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_1718.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf791.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf793.data_ptr()))
    del convolution_3
    del primals_9
    del relu_4
    del squeeze_13
    del unsqueeze_1718
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf795 = aten.convolution_backward(buf794, relu_3, primals_226, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False])
    del buf794
    del primals_226
    buf796 = buf795[0]
    buf797 = buf795[1]
    del buf795
    buf798 = buf792; del buf792  # reuse
    buf799 = empty((200, ), device='cpu', dtype=torch.float32)
    buf800 = empty((200, ), device='cpu', dtype=torch.float32)
    buf801 = buf796; del buf796  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_110(c_void_p(buf801.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_1730.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf800.data_ptr()))
    del buf799
    del convolution_2
    del primals_7
    del relu_3
    del squeeze_10
    del unsqueeze_1730
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf802 = aten.convolution_backward(buf801, relu_2, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf801
    del primals_225
    buf803 = buf802[0]
    buf804 = buf802[1]
    del buf802
    buf808 = empty((8, 296, 56, 56), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_slice_backward_111(c_void_p(buf720.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf785.data_ptr()), c_void_p(buf808.data_ptr()))
    del buf720
    del buf741
    del buf763
    del buf785
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.slice_backward]
    buf809 = aten.convolution_backward(buf808, relu_1, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf808
    del primals_224
    buf810 = buf809[0]
    buf805 = empty((128, ), device='cpu', dtype=torch.float32)
    buf806 = empty((128, ), device='cpu', dtype=torch.float32)
    buf812 = empty((128, ), device='cpu', dtype=torch.float32)
    buf813 = empty((128, ), device='cpu', dtype=torch.float32)
    buf807 = empty((128, ), device='cpu', dtype=torch.float32)
    buf814 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_batch_norm_backward_threshold_backward_112(c_void_p(relu_2.data_ptr()), c_void_p(buf803.data_ptr()), c_void_p(sub_543.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf814.data_ptr()))
    buf811 = buf809[1]
    del buf809
    buf815 = buf803; del buf803  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_113(c_void_p(buf815.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(sub_543.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(primals_3.data_ptr()))
    del buf810
    del primals_3
    del primals_5
    del relu_1
    del relu_2
    del squeeze_4
    del sub_543
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf816 = aten.max_pool2d_with_indices_backward(buf815, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3)
    del buf815
    del getitem_3
    buf817 = buf816
    del buf816
    buf818 = buf813; del buf813  # reuse
    buf819 = buf806; del buf806  # reuse
    buf820 = empty((128, ), device='cpu', dtype=torch.float32)
    buf821 = buf817; del buf817  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_114(c_void_p(buf821.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_1766.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(buf819.data_ptr()), c_void_p(buf820.data_ptr()))
    del buf819
    del convolution
    del primals_1
    del relu
    del squeeze_1
    del unsqueeze_1766
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf822 = aten.convolution_backward(buf821, primals_668, primals_223, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf821
    del primals_223
    del primals_668
    buf823 = buf822[1]
    return (buf820, buf818, buf814, buf812, buf807, buf805, buf800, buf798, buf793, buf791, buf786, buf783, buf778, buf776, buf771, buf769, buf764, buf761, buf756, buf754, buf749, buf747, buf742, buf739, buf734, buf732, buf727, buf725, buf719, buf717, buf712, buf710, buf705, buf703, buf698, buf696, buf691, buf688, buf683, buf681, buf676, buf674, buf669, buf666, buf661, buf659, buf654, buf652, buf647, buf644, buf639, buf637, buf632, buf630, buf623, buf620, buf615, buf613, buf608, buf606, buf601, buf598, buf593, buf591, buf586, buf584, buf579, buf576, buf571, buf569, buf564, buf562, buf557, buf554, buf549, buf547, buf542, buf540, buf534, buf532, buf527, buf525, buf520, buf518, buf513, buf511, buf506, buf503, buf498, buf496, buf491, buf489, buf484, buf481, buf476, buf474, buf469, buf467, buf462, buf459, buf454, buf452, buf447, buf445, buf438, buf435, buf430, buf428, buf423, buf421, buf416, buf413, buf408, buf406, buf401, buf399, buf394, buf391, buf386, buf384, buf379, buf377, buf372, buf369, buf364, buf362, buf357, buf355, buf348, buf345, buf340, buf338, buf333, buf331, buf326, buf323, buf318, buf316, buf311, buf309, buf304, buf301, buf296, buf294, buf289, buf287, buf282, buf279, buf274, buf272, buf267, buf265, buf258, buf255, buf250, buf248, buf243, buf241, buf236, buf233, buf228, buf226, buf221, buf219, buf214, buf211, buf206, buf204, buf199, buf197, buf192, buf189, buf184, buf182, buf177, buf175, buf168, buf165, buf160, buf158, buf153, buf151, buf146, buf143, buf138, buf136, buf131, buf129, buf124, buf121, buf116, buf114, buf109, buf107, buf102, buf99, buf94, buf92, buf87, buf85, buf79, buf77, buf72, buf70, buf65, buf63, buf58, buf56, buf51, buf48, buf43, buf41, buf36, buf34, buf29, buf26, buf21, buf19, buf14, buf12, buf7, buf4, buf823, buf811, buf804, buf797, buf790, buf782, buf775, buf768, buf760, buf753, buf746, buf738, buf731, buf724, buf716, buf709, buf702, buf695, buf687, buf680, buf673, buf665, buf658, buf651, buf643, buf636, buf629, buf619, buf612, buf605, buf597, buf590, buf583, buf575, buf568, buf561, buf553, buf546, buf539, buf531, buf524, buf517, buf510, buf502, buf495, buf488, buf480, buf473, buf466, buf458, buf451, buf444, buf434, buf427, buf420, buf412, buf405, buf398, buf390, buf383, buf376, buf368, buf361, buf354, buf344, buf337, buf330, buf322, buf315, buf308, buf300, buf293, buf286, buf278, buf271, buf264, buf254, buf247, buf240, buf232, buf225, buf218, buf210, buf203, buf196, buf188, buf181, buf174, buf164, buf157, buf150, buf142, buf135, buf128, buf120, buf113, buf106, buf98, buf91, buf84, buf76, buf69, buf62, buf55, buf47, buf40, buf33, buf25, buf18, buf11, buf2, buf3, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((316, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((356, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((1088, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((1216, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1472, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((1664, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((1728, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((1792, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((1856, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((2112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((2176, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((2368, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((2688, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((128, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((296, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((200, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((200, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((200, 316, 1, 1), (316, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((200, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((200, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((200, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((200, 356, 1, 1), (356, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((200, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((640, 376, 1, 1), (376, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((400, 376, 1, 1), (376, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((400, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((400, 704, 1, 1), (704, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((400, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((400, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((400, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((400, 832, 1, 1), (832, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((400, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((400, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((400, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((400, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((400, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((400, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((400, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((400, 1088, 1, 1), (1088, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((400, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((1152, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((800, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((800, 1216, 1, 1), (1216, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((800, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((800, 1344, 1, 1), (1344, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((800, 1408, 1, 1), (1408, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((800, 1472, 1, 1), (1472, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((800, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((800, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((800, 1664, 1, 1), (1664, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((800, 1728, 1, 1), (1728, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((800, 1792, 1, 1), (1792, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((800, 1856, 1, 1), (1856, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((800, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((800, 1984, 1, 1), (1984, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((800, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((800, 2112, 1, 1), (2112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((800, 2176, 1, 1), (2176, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((800, 2240, 1, 1), (2240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((800, 2304, 1, 1), (2304, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((800, 2368, 1, 1), (2368, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((800, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((2304, 2432, 1, 1), (2432, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((1600, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((1600, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((1600, 2560, 1, 1), (2560, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((1600, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((1000, 2688, 1, 1), (2688, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_668 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 128, 112, 112), (1605632, 1, 14336, 128), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 128, 112, 112), (1605632, 1, 14336, 128), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.int64)
    squeeze_4 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 316, 56, 56), (990976, 1, 17696, 316), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((316, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 316, 56, 56), (990976, 1, 17696, 316), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((8, 336, 56, 56), (1053696, 1, 18816, 336), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 336, 56, 56), (1053696, 1, 18816, 336), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    cat_5 = rand_strided((8, 356, 56, 56), (1116416, 1, 19936, 356), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((356, ), (1, ), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 356, 56, 56), (1116416, 1, 19936, 356), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 200, 56, 56), (627200, 1, 11200, 200), device='cpu', dtype=torch.float32)
    cat_7 = rand_strided((8, 376, 56, 56), (1179136, 1, 21056, 376), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((376, ), (1, ), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((8, 376, 56, 56), (1179136, 1, 21056, 376), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 376, 56, 56), (1179136, 1, 21056, 376), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 400, 56, 56), (1254400, 1, 22400, 400), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((8, 400, 56, 56), (1254400, 1, 22400, 400), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    cat_9 = rand_strided((8, 704, 28, 28), (551936, 1, 19712, 704), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((704, ), (1, ), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((8, 704, 28, 28), (551936, 1, 19712, 704), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    cat_11 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    cat_13 = rand_strided((8, 832, 28, 28), (652288, 1, 23296, 832), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((832, ), (1, ), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((8, 832, 28, 28), (652288, 1, 23296, 832), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    cat_15 = rand_strided((8, 896, 28, 28), (702464, 1, 25088, 896), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((8, 896, 28, 28), (702464, 1, 25088, 896), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    cat_17 = rand_strided((8, 960, 28, 28), (752640, 1, 26880, 960), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((8, 960, 28, 28), (752640, 1, 26880, 960), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    cat_19 = rand_strided((8, 1024, 28, 28), (802816, 1, 28672, 1024), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((8, 1024, 28, 28), (802816, 1, 28672, 1024), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    cat_21 = rand_strided((8, 1088, 28, 28), (852992, 1, 30464, 1088), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((1088, ), (1, ), device='cpu', dtype=torch.float32)
    relu_36 = rand_strided((8, 1088, 28, 28), (852992, 1, 30464, 1088), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((400, ), (1, ), device='cpu', dtype=torch.float32)
    relu_38 = rand_strided((8, 400, 28, 28), (313600, 1, 11200, 400), device='cpu', dtype=torch.float32)
    cat_23 = rand_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    relu_39 = rand_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cpu', dtype=torch.float32)
    relu_40 = rand_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 800, 28, 28), (627200, 1, 22400, 800), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_41 = rand_strided((8, 800, 28, 28), (627200, 1, 22400, 800), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_42 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_25 = rand_strided((8, 1216, 14, 14), (238336, 1, 17024, 1216), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((1216, ), (1, ), device='cpu', dtype=torch.float32)
    relu_43 = rand_strided((8, 1216, 14, 14), (238336, 1, 17024, 1216), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_44 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_27 = rand_strided((8, 1280, 14, 14), (250880, 1, 17920, 1280), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    relu_46 = rand_strided((8, 1280, 14, 14), (250880, 1, 17920, 1280), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_47 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_48 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_29 = rand_strided((8, 1344, 14, 14), (263424, 1, 18816, 1344), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((1344, ), (1, ), device='cpu', dtype=torch.float32)
    relu_49 = rand_strided((8, 1344, 14, 14), (263424, 1, 18816, 1344), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_50 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_51 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_31 = rand_strided((8, 1408, 14, 14), (275968, 1, 19712, 1408), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    relu_52 = rand_strided((8, 1408, 14, 14), (275968, 1, 19712, 1408), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_53 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_54 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_33 = rand_strided((8, 1472, 14, 14), (288512, 1, 20608, 1472), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((1472, ), (1, ), device='cpu', dtype=torch.float32)
    relu_55 = rand_strided((8, 1472, 14, 14), (288512, 1, 20608, 1472), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_56 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_57 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_35 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    relu_58 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_59 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_60 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_37 = rand_strided((8, 1600, 14, 14), (313600, 1, 22400, 1600), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    relu_61 = rand_strided((8, 1600, 14, 14), (313600, 1, 22400, 1600), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_187 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_62 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_190 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_63 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_39 = rand_strided((8, 1664, 14, 14), (326144, 1, 23296, 1664), device='cpu', dtype=torch.float32)
    squeeze_193 = rand_strided((1664, ), (1, ), device='cpu', dtype=torch.float32)
    relu_64 = rand_strided((8, 1664, 14, 14), (326144, 1, 23296, 1664), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_196 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_65 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_199 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_66 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_41 = rand_strided((8, 1728, 14, 14), (338688, 1, 24192, 1728), device='cpu', dtype=torch.float32)
    squeeze_202 = rand_strided((1728, ), (1, ), device='cpu', dtype=torch.float32)
    relu_67 = rand_strided((8, 1728, 14, 14), (338688, 1, 24192, 1728), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_205 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_68 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_208 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_69 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_43 = rand_strided((8, 1792, 14, 14), (351232, 1, 25088, 1792), device='cpu', dtype=torch.float32)
    squeeze_211 = rand_strided((1792, ), (1, ), device='cpu', dtype=torch.float32)
    relu_70 = rand_strided((8, 1792, 14, 14), (351232, 1, 25088, 1792), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_214 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_71 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_217 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_72 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_45 = rand_strided((8, 1856, 14, 14), (363776, 1, 25984, 1856), device='cpu', dtype=torch.float32)
    squeeze_220 = rand_strided((1856, ), (1, ), device='cpu', dtype=torch.float32)
    relu_73 = rand_strided((8, 1856, 14, 14), (363776, 1, 25984, 1856), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_223 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_74 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_226 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_75 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_47 = rand_strided((8, 1920, 14, 14), (376320, 1, 26880, 1920), device='cpu', dtype=torch.float32)
    squeeze_229 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_76 = rand_strided((8, 1920, 14, 14), (376320, 1, 26880, 1920), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_232 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_77 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_235 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_78 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_49 = rand_strided((8, 1984, 14, 14), (388864, 1, 27776, 1984), device='cpu', dtype=torch.float32)
    squeeze_238 = rand_strided((1984, ), (1, ), device='cpu', dtype=torch.float32)
    relu_79 = rand_strided((8, 1984, 14, 14), (388864, 1, 27776, 1984), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_241 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_80 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_244 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_81 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_51 = rand_strided((8, 2048, 14, 14), (401408, 1, 28672, 2048), device='cpu', dtype=torch.float32)
    squeeze_247 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    relu_82 = rand_strided((8, 2048, 14, 14), (401408, 1, 28672, 2048), device='cpu', dtype=torch.float32)
    convolution_82 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_250 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_83 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_83 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_253 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_84 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_53 = rand_strided((8, 2112, 14, 14), (413952, 1, 29568, 2112), device='cpu', dtype=torch.float32)
    squeeze_256 = rand_strided((2112, ), (1, ), device='cpu', dtype=torch.float32)
    relu_85 = rand_strided((8, 2112, 14, 14), (413952, 1, 29568, 2112), device='cpu', dtype=torch.float32)
    convolution_85 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_259 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_86 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_86 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_262 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_87 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_55 = rand_strided((8, 2176, 14, 14), (426496, 1, 30464, 2176), device='cpu', dtype=torch.float32)
    squeeze_265 = rand_strided((2176, ), (1, ), device='cpu', dtype=torch.float32)
    relu_88 = rand_strided((8, 2176, 14, 14), (426496, 1, 30464, 2176), device='cpu', dtype=torch.float32)
    convolution_88 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_268 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_89 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_89 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_271 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_90 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_57 = rand_strided((8, 2240, 14, 14), (439040, 1, 31360, 2240), device='cpu', dtype=torch.float32)
    squeeze_274 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    relu_91 = rand_strided((8, 2240, 14, 14), (439040, 1, 31360, 2240), device='cpu', dtype=torch.float32)
    convolution_91 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_277 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_92 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_92 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_280 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_93 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_59 = rand_strided((8, 2304, 14, 14), (451584, 1, 32256, 2304), device='cpu', dtype=torch.float32)
    squeeze_283 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    relu_94 = rand_strided((8, 2304, 14, 14), (451584, 1, 32256, 2304), device='cpu', dtype=torch.float32)
    convolution_94 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_286 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_95 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_95 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_289 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_96 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_61 = rand_strided((8, 2368, 14, 14), (464128, 1, 33152, 2368), device='cpu', dtype=torch.float32)
    squeeze_292 = rand_strided((2368, ), (1, ), device='cpu', dtype=torch.float32)
    relu_97 = rand_strided((8, 2368, 14, 14), (464128, 1, 33152, 2368), device='cpu', dtype=torch.float32)
    convolution_97 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_295 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_98 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    convolution_98 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    squeeze_298 = rand_strided((800, ), (1, ), device='cpu', dtype=torch.float32)
    relu_99 = rand_strided((8, 800, 14, 14), (156800, 1, 11200, 800), device='cpu', dtype=torch.float32)
    cat_63 = rand_strided((8, 2432, 14, 14), (476672, 1, 34048, 2432), device='cpu', dtype=torch.float32)
    squeeze_301 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    relu_100 = rand_strided((8, 2432, 14, 14), (476672, 1, 34048, 2432), device='cpu', dtype=torch.float32)
    relu_101 = rand_strided((8, 2432, 14, 14), (476672, 1, 34048, 2432), device='cpu', dtype=torch.float32)
    convolution_101 = rand_strided((8, 1600, 14, 14), (313600, 1, 22400, 1600), device='cpu', dtype=torch.float32)
    squeeze_307 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    relu_102 = rand_strided((8, 1600, 14, 14), (313600, 1, 22400, 1600), device='cpu', dtype=torch.float32)
    convolution_102 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    squeeze_310 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    relu_103 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    cat_65 = rand_strided((8, 2432, 7, 7), (119168, 1, 17024, 2432), device='cpu', dtype=torch.float32)
    squeeze_313 = rand_strided((2432, ), (1, ), device='cpu', dtype=torch.float32)
    relu_104 = rand_strided((8, 2432, 7, 7), (119168, 1, 17024, 2432), device='cpu', dtype=torch.float32)
    convolution_104 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    squeeze_316 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    relu_105 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    convolution_105 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    squeeze_319 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    relu_106 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    cat_67 = rand_strided((8, 2560, 7, 7), (125440, 1, 17920, 2560), device='cpu', dtype=torch.float32)
    squeeze_322 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    relu_107 = rand_strided((8, 2560, 7, 7), (125440, 1, 17920, 2560), device='cpu', dtype=torch.float32)
    convolution_107 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    squeeze_325 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    relu_108 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    convolution_108 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    squeeze_328 = rand_strided((1600, ), (1, ), device='cpu', dtype=torch.float32)
    relu_109 = rand_strided((8, 1600, 7, 7), (78400, 1, 11200, 1600), device='cpu', dtype=torch.float32)
    cat_69 = rand_strided((8, 2688, 7, 7), (131712, 1, 18816, 2688), device='cpu', dtype=torch.float32)
    squeeze_331 = rand_strided((2688, ), (1, ), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 2688, 1, 1), (2688, 1, 2688, 2688), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 2688, 7, 7), (131712, 1, 18816, 2688), device='cpu', dtype=torch.bool)
    unsqueeze_446 = rand_strided((1, 2688, 1, 1), (2688, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_470 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 2560, 1, 1), (2560, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_494 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_518 = rand_strided((1, 2432, 1, 1), (2432, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_530 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_542 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_554 = rand_strided((1, 2432, 1, 1), (2432, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_578 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_590 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_602 = rand_strided((1, 2368, 1, 1), (2368, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_614 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_626 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_638 = rand_strided((1, 2304, 1, 1), (2304, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_650 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_662 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_674 = rand_strided((1, 2240, 1, 1), (2240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_686 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_698 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_710 = rand_strided((1, 2176, 1, 1), (2176, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_722 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_734 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 2112, 1, 1), (2112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_770 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_782 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_794 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_806 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_818 = rand_strided((1, 1984, 1, 1), (1984, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_830 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_842 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_854 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_866 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_878 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_890 = rand_strided((1, 1856, 1, 1), (1856, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_902 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_914 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_926 = rand_strided((1, 1792, 1, 1), (1792, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_938 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_950 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_962 = rand_strided((1, 1728, 1, 1), (1728, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_974 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_986 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_998 = rand_strided((1, 1664, 1, 1), (1664, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1010 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1022 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1034 = rand_strided((1, 1600, 1, 1), (1600, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1046 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1058 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1070 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1082 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1094 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1106 = rand_strided((1, 1472, 1, 1), (1472, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1118 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1130 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1142 = rand_strided((1, 1408, 1, 1), (1408, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1154 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1166 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1178 = rand_strided((1, 1344, 1, 1), (1344, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1190 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1202 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1214 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1226 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1238 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1250 = rand_strided((1, 1216, 1, 1), (1216, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1262 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1274 = rand_strided((1, 800, 1, 1), (800, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1286 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1310 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1322 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1334 = rand_strided((1, 1088, 1, 1), (1088, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1346 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1358 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1370 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1382 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1394 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1406 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1418 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1430 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1442 = rand_strided((1, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1454 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1466 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1478 = rand_strided((1, 832, 1, 1), (832, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1490 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1502 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1514 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1526 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1538 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1550 = rand_strided((1, 704, 1, 1), (704, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1562 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1574 = rand_strided((1, 400, 1, 1), (400, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1586 = rand_strided((1, 376, 1, 1), (376, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1610 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1622 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1634 = rand_strided((1, 356, 1, 1), (356, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1646 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1658 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1670 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1682 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1694 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1706 = rand_strided((1, 316, 1, 1), (316, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1718 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1730 = rand_strided((1, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    sub_543 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    unsqueeze_1766 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_668, convolution, squeeze_1, relu, getitem_3, squeeze_4, relu_1, relu_2, convolution_2, squeeze_10, relu_3, convolution_3, squeeze_13, relu_4, cat_1, squeeze_16, relu_5, convolution_5, squeeze_19, relu_6, convolution_6, squeeze_22, relu_7, cat_3, squeeze_25, relu_8, convolution_8, squeeze_28, relu_9, convolution_9, squeeze_31, relu_10, cat_5, squeeze_34, relu_11, convolution_11, squeeze_37, relu_12, convolution_12, squeeze_40, relu_13, cat_7, squeeze_43, relu_14, relu_15, convolution_15, squeeze_49, relu_16, convolution_16, squeeze_52, relu_17, cat_9, squeeze_55, relu_18, convolution_18, squeeze_58, relu_19, convolution_19, squeeze_61, relu_20, cat_11, squeeze_64, relu_21, convolution_21, squeeze_67, relu_22, convolution_22, squeeze_70, relu_23, cat_13, squeeze_73, relu_24, convolution_24, squeeze_76, relu_25, convolution_25, squeeze_79, relu_26, cat_15, squeeze_82, relu_27, convolution_27, squeeze_85, relu_28, convolution_28, squeeze_88, relu_29, cat_17, squeeze_91, relu_30, convolution_30, squeeze_94, relu_31, convolution_31, squeeze_97, relu_32, cat_19, squeeze_100, relu_33, convolution_33, squeeze_103, relu_34, convolution_34, squeeze_106, relu_35, cat_21, squeeze_109, relu_36, convolution_36, squeeze_112, relu_37, convolution_37, squeeze_115, relu_38, cat_23, squeeze_118, relu_39, relu_40, convolution_40, squeeze_124, relu_41, convolution_41, squeeze_127, relu_42, cat_25, squeeze_130, relu_43, convolution_43, squeeze_133, relu_44, convolution_44, squeeze_136, relu_45, cat_27, squeeze_139, relu_46, convolution_46, squeeze_142, relu_47, convolution_47, squeeze_145, relu_48, cat_29, squeeze_148, relu_49, convolution_49, squeeze_151, relu_50, convolution_50, squeeze_154, relu_51, cat_31, squeeze_157, relu_52, convolution_52, squeeze_160, relu_53, convolution_53, squeeze_163, relu_54, cat_33, squeeze_166, relu_55, convolution_55, squeeze_169, relu_56, convolution_56, squeeze_172, relu_57, cat_35, squeeze_175, relu_58, convolution_58, squeeze_178, relu_59, convolution_59, squeeze_181, relu_60, cat_37, squeeze_184, relu_61, convolution_61, squeeze_187, relu_62, convolution_62, squeeze_190, relu_63, cat_39, squeeze_193, relu_64, convolution_64, squeeze_196, relu_65, convolution_65, squeeze_199, relu_66, cat_41, squeeze_202, relu_67, convolution_67, squeeze_205, relu_68, convolution_68, squeeze_208, relu_69, cat_43, squeeze_211, relu_70, convolution_70, squeeze_214, relu_71, convolution_71, squeeze_217, relu_72, cat_45, squeeze_220, relu_73, convolution_73, squeeze_223, relu_74, convolution_74, squeeze_226, relu_75, cat_47, squeeze_229, relu_76, convolution_76, squeeze_232, relu_77, convolution_77, squeeze_235, relu_78, cat_49, squeeze_238, relu_79, convolution_79, squeeze_241, relu_80, convolution_80, squeeze_244, relu_81, cat_51, squeeze_247, relu_82, convolution_82, squeeze_250, relu_83, convolution_83, squeeze_253, relu_84, cat_53, squeeze_256, relu_85, convolution_85, squeeze_259, relu_86, convolution_86, squeeze_262, relu_87, cat_55, squeeze_265, relu_88, convolution_88, squeeze_268, relu_89, convolution_89, squeeze_271, relu_90, cat_57, squeeze_274, relu_91, convolution_91, squeeze_277, relu_92, convolution_92, squeeze_280, relu_93, cat_59, squeeze_283, relu_94, convolution_94, squeeze_286, relu_95, convolution_95, squeeze_289, relu_96, cat_61, squeeze_292, relu_97, convolution_97, squeeze_295, relu_98, convolution_98, squeeze_298, relu_99, cat_63, squeeze_301, relu_100, relu_101, convolution_101, squeeze_307, relu_102, convolution_102, squeeze_310, relu_103, cat_65, squeeze_313, relu_104, convolution_104, squeeze_316, relu_105, convolution_105, squeeze_319, relu_106, cat_67, squeeze_322, relu_107, convolution_107, squeeze_325, relu_108, convolution_108, squeeze_328, relu_109, cat_69, squeeze_331, mean, le, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, unsqueeze_1022, unsqueeze_1034, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, unsqueeze_1094, unsqueeze_1106, unsqueeze_1118, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1382, unsqueeze_1394, unsqueeze_1406, unsqueeze_1418, unsqueeze_1430, unsqueeze_1442, unsqueeze_1454, unsqueeze_1466, unsqueeze_1478, unsqueeze_1490, unsqueeze_1502, unsqueeze_1514, unsqueeze_1526, unsqueeze_1538, unsqueeze_1550, unsqueeze_1562, unsqueeze_1574, unsqueeze_1586, unsqueeze_1610, unsqueeze_1622, unsqueeze_1634, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, unsqueeze_1682, unsqueeze_1694, unsqueeze_1706, unsqueeze_1718, unsqueeze_1730, sub_543, unsqueeze_1766, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dpn107', benchmark_compiled_module)
