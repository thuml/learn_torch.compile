
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


cpp_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_0 = async_compile.cpp('''
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
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
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const bool* in_ptr0,
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp1 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2560L*x2) + (125440L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp3 = static_cast<float>(49.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 / tmp4;
                            auto tmp6 = static_cast<float>(0.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp1);
                            auto tmp10 = tmp8 + tmp9;
                            auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp0);
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp15 = tmp11 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp1 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2560L*x1) + (125440L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp3 = static_cast<float>(49.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 / tmp4;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp1);
                        auto tmp10 = tmp8 + tmp9;
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp0);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp16 = static_cast<float>(0.002551020408163265);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = tmp19 * tmp19;
                        auto tmp21 = tmp18 * tmp20;
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp23 = tmp11 - tmp22;
                        auto tmp25 = tmp24 * tmp17;
                        auto tmp26 = tmp23 - tmp25;
                        tmp26.store(out_ptr2 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const bool* in_ptr3,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x2 + (2560L*x1) + (125440L*x0)));
                        auto tmp5 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp6 = flag_to_float_vec(in_ptr3 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (2560L*x1) + (125440L*x0)));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp8 = static_cast<float>(49.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = decltype(tmp2)::blendv(tmp10, tmp2, tmp6);
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp5);
                        auto tmp15 = tmp4 + tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp18 = decltype(tmp2)::blendv(tmp17, tmp2, tmp3);
                        tmp18.store(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                    }
                }
            }
        }
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_8 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2048L + x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2816L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = decltype(tmp3)::blendv(tmp7, tmp3, tmp4);
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp11 = decltype(tmp3)::blendv(tmp10, tmp3, tmp0);
                    tmp11.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (2816L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (2816L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(2304L + x1 + (2816L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (2816L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = to_float_mask(tmp5 <= tmp2);
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp6);
                    auto tmp13 = tmp4 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = decltype(tmp2)::blendv(tmp15, tmp2, tmp3);
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                       const bool* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1024L*x0)));
                    auto tmp5 = flag_to_float_vec(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1792L + x0 + (2816L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1792L + x1 + (2816L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1792L + x1 + (2816L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp10 = decltype(tmp3)::blendv(tmp9, tmp3, tmp4);
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp3)::blendv(tmp12, tmp3, tmp0);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (1536L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1536L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x1 + (1536L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1536L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = to_float_mask(tmp5 <= tmp2);
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp6);
                    auto tmp13 = tmp4 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = decltype(tmp2)::blendv(tmp15, tmp2, tmp3);
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1024L*x0)));
                    auto tmp5 = flag_to_float_vec(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1280L + x0 + (2816L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1280L + x1 + (2816L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1280L + x1 + (2816L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp10 = decltype(tmp3)::blendv(tmp9, tmp3, tmp4);
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp3)::blendv(tmp12, tmp3, tmp0);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1536L + x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (2048L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = to_float_mask(tmp5 <= tmp2);
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp6);
                    auto tmp13 = tmp4 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = decltype(tmp2)::blendv(tmp15, tmp2, tmp3);
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_48 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1024L*x0)));
                    auto tmp5 = flag_to_float_vec(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x0 + (2048L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x1 + (2048L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1024L + x1 + (2048L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp10 = decltype(tmp3)::blendv(tmp9, tmp3, tmp4);
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp3)::blendv(tmp12, tmp3, tmp0);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (1536L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1536L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x1 + (1536L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1536L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = to_float_mask(tmp5 <= tmp2);
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp6);
                    auto tmp13 = tmp4 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = decltype(tmp2)::blendv(tmp15, tmp2, tmp3);
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1024L*x0)));
                    auto tmp5 = flag_to_float_vec(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp10 = decltype(tmp3)::blendv(tmp9, tmp3, tmp4);
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp3)::blendv(tmp12, tmp3, tmp0);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (1152L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (1152L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_71 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_72 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(896L + x1 + (1152L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (1152L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = to_float_mask(tmp5 <= tmp2);
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp6);
                    auto tmp13 = tmp4 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = decltype(tmp2)::blendv(tmp15, tmp2, tmp3);
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                       const bool* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
                    auto tmp5 = flag_to_float_vec(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(640L + x0 + (1152L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(640L + x1 + (1152L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(640L + x1 + (1152L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp10 = decltype(tmp3)::blendv(tmp9, tmp3, tmp4);
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp3)::blendv(tmp12, tmp3, tmp0);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_85 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = to_float_mask(tmp5 <= tmp2);
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp6);
                    auto tmp13 = tmp4 + tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = decltype(tmp2)::blendv(tmp15, tmp2, tmp3);
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
                    auto tmp5 = flag_to_float_vec(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_native_batch_norm_backward_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_95 = async_compile.cpp('''
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp3);
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp13 = tmp9 * tmp12;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    auto tmp26 = tmp17 * tmp25;
                    auto tmp27 = tmp24 * tmp26;
                    tmp27.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp10 = decltype(tmp3)::blendv(tmp9, tmp3, tmp4);
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp13 = decltype(tmp3)::blendv(tmp12, tmp3, tmp0);
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_97 = async_compile.cpp('''
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_98 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp39 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
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
                    tmp38.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp41.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr2 + static_cast<long>(x0));
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


cpp_fused_native_batch_norm_backward_102 = async_compile.cpp('''
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_103 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(401408L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(2.4912308673469386e-06);
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_105 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(401408L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(2.4912308673469386e-06);
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, primals_314, primals_316, primals_633, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, cat, convolution_10, squeeze_31, relu_9, getitem_24, getitem_25, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_10, convolution_13, squeeze_40, relu_11, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, cat_1, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, cat_2, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, relu_24, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, relu_26, convolution_29, squeeze_88, relu_27, convolution_30, squeeze_91, relu_28, convolution_31, squeeze_94, cat_3, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, relu_31, convolution_34, squeeze_103, relu_32, convolution_35, squeeze_106, relu_33, convolution_36, squeeze_109, relu_34, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, cat_4, convolution_39, squeeze_118, relu_37, getitem_88, getitem_89, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, relu_42, convolution_46, squeeze_139, cat_5, convolution_47, squeeze_142, relu_44, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, relu_46, convolution_50, squeeze_151, relu_47, convolution_51, squeeze_154, relu_48, convolution_52, squeeze_157, relu_49, convolution_53, squeeze_160, cat_6, convolution_54, squeeze_163, relu_51, convolution_55, squeeze_166, relu_52, convolution_56, squeeze_169, relu_53, convolution_57, squeeze_172, relu_54, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, relu_56, convolution_60, squeeze_181, cat_7, convolution_61, squeeze_184, relu_58, convolution_62, squeeze_187, relu_59, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, relu_61, convolution_65, squeeze_196, relu_62, convolution_66, squeeze_199, relu_63, convolution_67, squeeze_202, cat_8, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, relu_66, convolution_70, squeeze_211, relu_67, convolution_71, squeeze_214, relu_68, convolution_72, squeeze_217, relu_69, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, cat_9, convolution_75, squeeze_226, relu_72, convolution_76, squeeze_229, relu_73, convolution_77, squeeze_232, relu_74, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, relu_76, convolution_80, squeeze_241, relu_77, convolution_81, squeeze_244, cat_10, convolution_82, squeeze_247, relu_79, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, relu_81, convolution_85, squeeze_256, relu_82, convolution_86, squeeze_259, relu_83, convolution_87, squeeze_262, relu_84, convolution_88, squeeze_265, cat_11, convolution_89, squeeze_268, relu_86, convolution_90, squeeze_271, relu_87, convolution_91, squeeze_274, relu_88, convolution_92, squeeze_277, relu_89, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, relu_91, convolution_95, squeeze_286, cat_12, convolution_96, squeeze_289, relu_93, getitem_210, getitem_211, convolution_97, squeeze_292, convolution_98, squeeze_295, relu_94, convolution_99, squeeze_298, relu_95, convolution_100, squeeze_301, relu_96, convolution_101, squeeze_304, relu_97, convolution_102, squeeze_307, relu_98, convolution_103, squeeze_310, cat_13, convolution_104, squeeze_313, clone, le, unsqueeze_422, le_1, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, le_8, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, le_15, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, le_22, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, le_29, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, le_36, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, le_43, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, unsqueeze_1022, le_50, unsqueeze_1034, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, unsqueeze_1094, unsqueeze_1106, le_57, unsqueeze_1118, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, le_64, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, le_71, unsqueeze_1298, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, le_78, unsqueeze_1382, unsqueeze_1394, unsqueeze_1406, unsqueeze_1418, unsqueeze_1430, unsqueeze_1442, unsqueeze_1454, le_85, unsqueeze_1466, unsqueeze_1478, unsqueeze_1490, unsqueeze_1502, unsqueeze_1514, unsqueeze_1526, unsqueeze_1538, unsqueeze_1550, le_92, unsqueeze_1562, unsqueeze_1574, unsqueeze_1586, unsqueeze_1598, unsqueeze_1610, unsqueeze_1622, unsqueeze_1634, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_4, (16, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (32, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_10, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_22, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_25, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_28, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_34, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_40, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_43, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_46, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_52, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_55, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_58, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_61, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_64, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_67, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_70, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_73, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_79, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_82, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_85, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_88, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_91, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_92, (128, ), (1, ))
    assert_size_stride(primals_94, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_100, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_103, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_106, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_109, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_112, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_115, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_118, (256, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_121, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_124, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_127, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_130, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_133, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_136, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_139, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_140, (512, ), (1, ))
    assert_size_stride(primals_142, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_145, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_148, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_151, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_154, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_155, (256, ), (1, ))
    assert_size_stride(primals_157, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_160, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_161, (512, ), (1, ))
    assert_size_stride(primals_163, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_166, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_167, (256, ), (1, ))
    assert_size_stride(primals_169, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_170, (256, ), (1, ))
    assert_size_stride(primals_172, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_175, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_178, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_179, (256, ), (1, ))
    assert_size_stride(primals_181, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_182, (512, ), (1, ))
    assert_size_stride(primals_184, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_185, (512, ), (1, ))
    assert_size_stride(primals_187, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_188, (256, ), (1, ))
    assert_size_stride(primals_190, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_191, (256, ), (1, ))
    assert_size_stride(primals_193, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_196, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_197, (256, ), (1, ))
    assert_size_stride(primals_199, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_202, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_205, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_208, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_209, (256, ), (1, ))
    assert_size_stride(primals_211, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_214, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_217, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_220, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_221, (256, ), (1, ))
    assert_size_stride(primals_223, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_224, (512, ), (1, ))
    assert_size_stride(primals_226, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_227, (512, ), (1, ))
    assert_size_stride(primals_229, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_232, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_235, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_236, (512, ), (1, ))
    assert_size_stride(primals_238, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_241, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_244, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_247, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_248, (512, ), (1, ))
    assert_size_stride(primals_250, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_253, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_256, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_257, (512, ), (1, ))
    assert_size_stride(primals_259, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_262, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_265, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_266, (512, ), (1, ))
    assert_size_stride(primals_268, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_269, (512, ), (1, ))
    assert_size_stride(primals_271, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_272, (256, ), (1, ))
    assert_size_stride(primals_274, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_275, (256, ), (1, ))
    assert_size_stride(primals_277, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_278, (512, ), (1, ))
    assert_size_stride(primals_280, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_281, (256, ), (1, ))
    assert_size_stride(primals_283, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_286, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_289, (512, 2816, 1, 1), (2816, 1, 1, 1))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_292, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_293, (1024, ), (1, ))
    assert_size_stride(primals_295, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_296, (512, ), (1, ))
    assert_size_stride(primals_298, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_301, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_304, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_307, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_308, (512, ), (1, ))
    assert_size_stride(primals_310, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_316, (1000, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_633, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 16, 224, 224), (802816, 1, 3584, 16))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(relu, (8, 16, 224, 224), (802816, 1, 3584, 16))
    assert_size_stride(convolution_1, (8, 16, 224, 224), (802816, 1, 3584, 16))
    assert_size_stride(squeeze_4, (16, ), (1, ))
    assert_size_stride(relu_1, (8, 16, 224, 224), (802816, 1, 3584, 16))
    assert_size_stride(convolution_2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_7, (32, ), (1, ))
    assert_size_stride(relu_2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(getitem_6, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(getitem_7, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_3, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_10, (128, ), (1, ))
    assert_size_stride(convolution_4, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(relu_3, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_5, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(relu_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_6, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_19, (128, ), (1, ))
    assert_size_stride(relu_5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_7, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(relu_6, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_8, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_7, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_9, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_28, (128, ), (1, ))
    assert_size_stride(cat, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_10, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_31, (128, ), (1, ))
    assert_size_stride(relu_9, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(getitem_24, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(getitem_25, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_11, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_34, (256, ), (1, ))
    assert_size_stride(convolution_12, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(relu_10, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_13, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_40, (128, ), (1, ))
    assert_size_stride(relu_11, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_14, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_43, (256, ), (1, ))
    assert_size_stride(relu_12, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_15, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(relu_13, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_16, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_49, (128, ), (1, ))
    assert_size_stride(relu_14, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_17, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_52, (256, ), (1, ))
    assert_size_stride(cat_1, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_18, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_55, (256, ), (1, ))
    assert_size_stride(relu_16, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_19, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(relu_17, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_20, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_61, (128, ), (1, ))
    assert_size_stride(relu_18, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_21, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_64, (256, ), (1, ))
    assert_size_stride(relu_19, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_22, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(relu_20, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_23, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_70, (128, ), (1, ))
    assert_size_stride(relu_21, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_24, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(cat_2, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_25, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_76, (256, ), (1, ))
    assert_size_stride(relu_23, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_26, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_79, (128, ), (1, ))
    assert_size_stride(relu_24, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_27, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_82, (128, ), (1, ))
    assert_size_stride(relu_25, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_28, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(relu_26, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_29, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_88, (128, ), (1, ))
    assert_size_stride(relu_27, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_30, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_91, (128, ), (1, ))
    assert_size_stride(relu_28, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_31, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_94, (256, ), (1, ))
    assert_size_stride(cat_3, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_32, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_97, (256, ), (1, ))
    assert_size_stride(relu_30, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_33, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_100, (128, ), (1, ))
    assert_size_stride(relu_31, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_34, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_103, (128, ), (1, ))
    assert_size_stride(relu_32, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_35, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_106, (256, ), (1, ))
    assert_size_stride(relu_33, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_36, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_109, (128, ), (1, ))
    assert_size_stride(relu_34, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_37, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_112, (128, ), (1, ))
    assert_size_stride(relu_35, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_38, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(cat_4, (8, 1152, 28, 28), (903168, 1, 32256, 1152))
    assert_size_stride(convolution_39, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_118, (256, ), (1, ))
    assert_size_stride(relu_37, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_88, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(getitem_89, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_40, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_121, (512, ), (1, ))
    assert_size_stride(convolution_41, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_124, (256, ), (1, ))
    assert_size_stride(relu_38, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_42, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_127, (256, ), (1, ))
    assert_size_stride(relu_39, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_43, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_130, (512, ), (1, ))
    assert_size_stride(relu_40, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_44, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_133, (256, ), (1, ))
    assert_size_stride(relu_41, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_45, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_136, (256, ), (1, ))
    assert_size_stride(relu_42, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_46, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_139, (512, ), (1, ))
    assert_size_stride(cat_5, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_47, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_142, (512, ), (1, ))
    assert_size_stride(relu_44, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_48, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_145, (256, ), (1, ))
    assert_size_stride(relu_45, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_49, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_148, (256, ), (1, ))
    assert_size_stride(relu_46, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_50, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_151, (512, ), (1, ))
    assert_size_stride(relu_47, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_51, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_154, (256, ), (1, ))
    assert_size_stride(relu_48, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_52, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_157, (256, ), (1, ))
    assert_size_stride(relu_49, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_53, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_160, (512, ), (1, ))
    assert_size_stride(cat_6, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_54, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_163, (512, ), (1, ))
    assert_size_stride(relu_51, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_55, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_166, (256, ), (1, ))
    assert_size_stride(relu_52, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_56, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_169, (256, ), (1, ))
    assert_size_stride(relu_53, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_57, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_172, (512, ), (1, ))
    assert_size_stride(relu_54, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_58, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_175, (256, ), (1, ))
    assert_size_stride(relu_55, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_59, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_178, (256, ), (1, ))
    assert_size_stride(relu_56, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_60, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_181, (512, ), (1, ))
    assert_size_stride(cat_7, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_61, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_184, (512, ), (1, ))
    assert_size_stride(relu_58, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_62, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_187, (256, ), (1, ))
    assert_size_stride(relu_59, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_63, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_190, (256, ), (1, ))
    assert_size_stride(relu_60, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_64, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_193, (512, ), (1, ))
    assert_size_stride(relu_61, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_65, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_196, (256, ), (1, ))
    assert_size_stride(relu_62, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_66, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_199, (256, ), (1, ))
    assert_size_stride(relu_63, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_67, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_202, (512, ), (1, ))
    assert_size_stride(cat_8, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
    assert_size_stride(convolution_68, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_205, (512, ), (1, ))
    assert_size_stride(relu_65, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_69, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_208, (256, ), (1, ))
    assert_size_stride(relu_66, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_70, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_211, (256, ), (1, ))
    assert_size_stride(relu_67, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_71, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_214, (512, ), (1, ))
    assert_size_stride(relu_68, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_72, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_217, (256, ), (1, ))
    assert_size_stride(relu_69, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_73, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_220, (256, ), (1, ))
    assert_size_stride(relu_70, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_74, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_223, (512, ), (1, ))
    assert_size_stride(cat_9, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_75, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_226, (512, ), (1, ))
    assert_size_stride(relu_72, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_76, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_229, (256, ), (1, ))
    assert_size_stride(relu_73, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_77, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_232, (256, ), (1, ))
    assert_size_stride(relu_74, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_78, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_235, (512, ), (1, ))
    assert_size_stride(relu_75, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_79, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_238, (256, ), (1, ))
    assert_size_stride(relu_76, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_80, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_241, (256, ), (1, ))
    assert_size_stride(relu_77, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_81, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_244, (512, ), (1, ))
    assert_size_stride(cat_10, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_82, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_247, (512, ), (1, ))
    assert_size_stride(relu_79, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_83, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_250, (256, ), (1, ))
    assert_size_stride(relu_80, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_84, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_253, (256, ), (1, ))
    assert_size_stride(relu_81, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_85, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_256, (512, ), (1, ))
    assert_size_stride(relu_82, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_86, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_259, (256, ), (1, ))
    assert_size_stride(relu_83, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_87, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_262, (256, ), (1, ))
    assert_size_stride(relu_84, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_88, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_265, (512, ), (1, ))
    assert_size_stride(cat_11, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_89, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_268, (512, ), (1, ))
    assert_size_stride(relu_86, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_90, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_271, (256, ), (1, ))
    assert_size_stride(relu_87, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_91, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_274, (256, ), (1, ))
    assert_size_stride(relu_88, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_92, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_277, (512, ), (1, ))
    assert_size_stride(relu_89, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_93, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_280, (256, ), (1, ))
    assert_size_stride(relu_90, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_94, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_283, (256, ), (1, ))
    assert_size_stride(relu_91, (8, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_95, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_286, (512, ), (1, ))
    assert_size_stride(cat_12, (8, 2816, 14, 14), (551936, 1, 39424, 2816))
    assert_size_stride(convolution_96, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_289, (512, ), (1, ))
    assert_size_stride(relu_93, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_210, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(getitem_211, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_97, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_292, (1024, ), (1, ))
    assert_size_stride(convolution_98, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_295, (512, ), (1, ))
    assert_size_stride(relu_94, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_99, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_298, (512, ), (1, ))
    assert_size_stride(relu_95, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_100, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_301, (1024, ), (1, ))
    assert_size_stride(relu_96, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_101, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_304, (512, ), (1, ))
    assert_size_stride(relu_97, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_102, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_307, (512, ), (1, ))
    assert_size_stride(relu_98, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_103, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_310, (1024, ), (1, ))
    assert_size_stride(cat_13, (8, 2560, 7, 7), (125440, 1, 17920, 2560))
    assert_size_stride(convolution_104, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_313, (1024, ), (1, ))
    assert_size_stride(clone, (8, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(le, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(unsqueeze_422, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_1, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(unsqueeze_434, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_446, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_470, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_494, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_518, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_8, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_530, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_542, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_554, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_566, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_578, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_590, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_602, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_15, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_614, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_626, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_638, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_650, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_662, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_674, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_686, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_22, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_698, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_710, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_722, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_734, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_746, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_758, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_770, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_29, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_782, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_794, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_806, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_818, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_830, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_842, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_854, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_36, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_866, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_878, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_890, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_902, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_914, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_926, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_938, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_43, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_950, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_962, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_974, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_986, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_998, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1010, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1022, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_50, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_1034, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1046, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1058, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1070, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1082, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1094, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1106, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_57, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_1118, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1130, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1142, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1154, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1166, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1178, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1190, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1202, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_64, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1214, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1226, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1238, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1250, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1262, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1274, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1286, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_71, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1298, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1310, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1322, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1334, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1346, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1358, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1370, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_78, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1382, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1394, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1406, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1418, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1430, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1442, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1454, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_85, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1466, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1478, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1490, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1502, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1514, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1526, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1538, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1550, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_92, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(unsqueeze_1562, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1574, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1586, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1598, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1610, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1622, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1634, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1646, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_1658, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1670, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf0 = aten.convolution_backward(reinterpret_tensor(tangents_1, (8, 1000, 1, 1), (1000, 1, 1, 1), 0), clone, primals_316, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del clone
    del primals_316
    del tangents_1
    buf1 = buf0[0]
    buf2 = buf0[1]
    buf3 = buf0[2]
    del buf0
    buf4 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf5 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf6 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_0(c_void_p(le.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(convolution_104.data_ptr()), c_void_p(unsqueeze_422.data_ptr()), c_void_p(squeeze_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del convolution_104
    del primals_314
    del squeeze_313
    del unsqueeze_422
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf8 = aten.convolution_backward(buf7, cat_13, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_13
    del primals_313
    buf9 = buf8[0]
    buf10 = buf8[1]
    del buf8
    buf11 = buf5; del buf5  # reuse
    buf12 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf13 = buf7; del buf7  # reuse
    buf14 = buf12; del buf12  # reuse
    buf15 = buf13; del buf13  # reuse
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_1(c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(le_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(convolution_103.data_ptr()), c_void_p(unsqueeze_434.data_ptr()), c_void_p(squeeze_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(buf11.data_ptr()))
    del convolution_103
    del primals_311
    del squeeze_310
    del unsqueeze_434
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf16 = aten.convolution_backward(buf15, relu_98, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_310
    buf17 = buf16[0]
    buf18 = buf16[1]
    del buf16
    buf19 = empty((512, ), device='cpu', dtype=torch.float32)
    buf20 = empty((512, ), device='cpu', dtype=torch.float32)
    buf21 = empty((512, ), device='cpu', dtype=torch.float32)
    buf22 = buf17; del buf17  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2(c_void_p(buf22.data_ptr()), c_void_p(relu_98.data_ptr()), c_void_p(convolution_102.data_ptr()), c_void_p(unsqueeze_446.data_ptr()), c_void_p(squeeze_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del convolution_102
    del primals_308
    del relu_98
    del squeeze_307
    del unsqueeze_446
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf23 = aten.convolution_backward(buf22, relu_97, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf22
    del primals_307
    buf24 = buf23[0]
    buf25 = buf23[1]
    del buf23
    buf26 = buf20; del buf20  # reuse
    buf27 = empty((512, ), device='cpu', dtype=torch.float32)
    buf28 = empty((512, ), device='cpu', dtype=torch.float32)
    buf29 = buf24; del buf24  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3(c_void_p(buf29.data_ptr()), c_void_p(relu_97.data_ptr()), c_void_p(convolution_101.data_ptr()), c_void_p(unsqueeze_458.data_ptr()), c_void_p(squeeze_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    del convolution_101
    del primals_305
    del relu_97
    del squeeze_304
    del unsqueeze_458
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf30 = aten.convolution_backward(buf29, relu_96, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf29
    del primals_304
    buf31 = buf30[0]
    buf32 = buf30[1]
    del buf30
    buf33 = buf31; del buf31  # reuse
    buf34 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf35 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf55 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf36 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf37 = buf15; del buf15  # reuse
    buf57 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_4(c_void_p(buf33.data_ptr()), c_void_p(relu_96.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(le_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(convolution_100.data_ptr()), c_void_p(unsqueeze_470.data_ptr()), c_void_p(convolution_97.data_ptr()), c_void_p(unsqueeze_506.data_ptr()), c_void_p(squeeze_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(squeeze_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf57.data_ptr()))
    del buf1
    del buf33
    del buf35
    del convolution_100
    del convolution_97
    del le
    del le_1
    del primals_293
    del primals_302
    del relu_96
    del squeeze_301
    del unsqueeze_470
    del unsqueeze_506
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf38 = aten.convolution_backward(buf37, relu_95, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf37
    del primals_301
    buf39 = buf38[0]
    buf40 = buf38[1]
    del buf38
    buf41 = buf27; del buf27  # reuse
    buf42 = empty((512, ), device='cpu', dtype=torch.float32)
    buf43 = empty((512, ), device='cpu', dtype=torch.float32)
    buf44 = buf39; del buf39  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(buf44.data_ptr()), c_void_p(relu_95.data_ptr()), c_void_p(convolution_99.data_ptr()), c_void_p(unsqueeze_482.data_ptr()), c_void_p(squeeze_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del convolution_99
    del primals_299
    del relu_95
    del squeeze_298
    del unsqueeze_482
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf45 = aten.convolution_backward(buf44, relu_94, primals_298, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf44
    del primals_298
    buf46 = buf45[0]
    buf47 = buf45[1]
    del buf45
    buf48 = buf42; del buf42  # reuse
    buf49 = empty((512, ), device='cpu', dtype=torch.float32)
    buf50 = empty((512, ), device='cpu', dtype=torch.float32)
    buf51 = buf46; del buf46  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(buf51.data_ptr()), c_void_p(relu_94.data_ptr()), c_void_p(convolution_98.data_ptr()), c_void_p(unsqueeze_494.data_ptr()), c_void_p(squeeze_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    del convolution_98
    del primals_296
    del relu_94
    del squeeze_295
    del unsqueeze_494
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf52 = aten.convolution_backward(buf51, relu_93, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_295
    buf53 = buf52[0]
    buf54 = buf52[1]
    del buf52
    buf56 = buf55; del buf55  # reuse
    cpp_fused_native_batch_norm_backward_7(c_void_p(buf56.data_ptr()), c_void_p(squeeze_292.data_ptr()))
    del squeeze_292
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf58 = aten.convolution_backward(buf57, getitem_210, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf57
    del getitem_210
    del primals_292
    buf59 = buf58[0]
    buf60 = buf58[1]
    del buf58
    buf61 = buf59; del buf59  # reuse
    cpp_fused_add_8(c_void_p(buf61.data_ptr()), c_void_p(buf9.data_ptr()))
    del buf9
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf62 = aten.max_pool2d_with_indices_backward(buf61, relu_93, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_211)
    del buf61
    del getitem_211
    buf63 = buf62
    del buf62
    buf64 = buf49; del buf49  # reuse
    buf65 = empty((512, ), device='cpu', dtype=torch.float32)
    buf66 = buf51; del buf51  # reuse
    buf67 = buf65; del buf65  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_9(c_void_p(buf67.data_ptr()), c_void_p(relu_93.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(convolution_96.data_ptr()), c_void_p(unsqueeze_518.data_ptr()), c_void_p(squeeze_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()))
    del convolution_96
    del primals_290
    del squeeze_289
    del unsqueeze_518
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf68 = aten.convolution_backward(buf66, cat_12, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_12
    del primals_289
    buf69 = buf68[0]
    buf70 = buf68[1]
    del buf68
    buf71 = buf53; del buf53  # reuse
    buf72 = empty((512, ), device='cpu', dtype=torch.float32)
    buf73 = empty((512, ), device='cpu', dtype=torch.float32)
    buf74 = empty((512, ), device='cpu', dtype=torch.float32)
    buf75 = buf66; del buf66  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf71.data_ptr()), c_void_p(le_8.data_ptr()), c_void_p(relu_93.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(convolution_95.data_ptr()), c_void_p(unsqueeze_530.data_ptr()), c_void_p(squeeze_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del buf63
    del convolution_95
    del le_8
    del primals_287
    del relu_93
    del squeeze_286
    del unsqueeze_530
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf76 = aten.convolution_backward(buf75, relu_91, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_286
    buf77 = buf76[0]
    buf78 = buf76[1]
    del buf76
    buf79 = empty((256, ), device='cpu', dtype=torch.float32)
    buf80 = empty((256, ), device='cpu', dtype=torch.float32)
    buf81 = empty((256, ), device='cpu', dtype=torch.float32)
    buf82 = buf77; del buf77  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(buf82.data_ptr()), c_void_p(relu_91.data_ptr()), c_void_p(convolution_94.data_ptr()), c_void_p(unsqueeze_542.data_ptr()), c_void_p(squeeze_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del convolution_94
    del primals_284
    del relu_91
    del squeeze_283
    del unsqueeze_542
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf83 = aten.convolution_backward(buf82, relu_90, primals_283, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf82
    del primals_283
    buf84 = buf83[0]
    buf85 = buf83[1]
    del buf83
    buf86 = buf80; del buf80  # reuse
    buf87 = empty((256, ), device='cpu', dtype=torch.float32)
    buf88 = empty((256, ), device='cpu', dtype=torch.float32)
    buf89 = buf84; del buf84  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf89.data_ptr()), c_void_p(relu_90.data_ptr()), c_void_p(convolution_93.data_ptr()), c_void_p(unsqueeze_554.data_ptr()), c_void_p(squeeze_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    del convolution_93
    del primals_281
    del relu_90
    del squeeze_280
    del unsqueeze_554
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf90 = aten.convolution_backward(buf89, relu_89, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf89
    del primals_280
    buf91 = buf90[0]
    buf92 = buf90[1]
    del buf90
    buf93 = buf73; del buf73  # reuse
    buf94 = empty((512, ), device='cpu', dtype=torch.float32)
    buf95 = buf75; del buf75  # reuse
    buf97 = buf95; del buf95  # reuse
    buf96 = buf94; del buf94  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(buf97.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(relu_89.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(convolution_92.data_ptr()), c_void_p(unsqueeze_566.data_ptr()), c_void_p(squeeze_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(buf93.data_ptr()))
    del convolution_92
    del primals_278
    del squeeze_277
    del unsqueeze_566
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf98 = aten.convolution_backward(buf97, relu_88, primals_277, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_277
    buf99 = buf98[0]
    buf100 = buf98[1]
    del buf98
    buf101 = buf87; del buf87  # reuse
    buf102 = empty((256, ), device='cpu', dtype=torch.float32)
    buf103 = empty((256, ), device='cpu', dtype=torch.float32)
    buf104 = buf99; del buf99  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf104.data_ptr()), c_void_p(relu_88.data_ptr()), c_void_p(convolution_91.data_ptr()), c_void_p(unsqueeze_578.data_ptr()), c_void_p(squeeze_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    del convolution_91
    del primals_275
    del relu_88
    del squeeze_274
    del unsqueeze_578
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf105 = aten.convolution_backward(buf104, relu_87, primals_274, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf104
    del primals_274
    buf106 = buf105[0]
    buf107 = buf105[1]
    del buf105
    buf108 = buf102; del buf102  # reuse
    buf109 = empty((256, ), device='cpu', dtype=torch.float32)
    buf110 = empty((256, ), device='cpu', dtype=torch.float32)
    buf111 = buf106; del buf106  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf111.data_ptr()), c_void_p(relu_87.data_ptr()), c_void_p(convolution_90.data_ptr()), c_void_p(unsqueeze_590.data_ptr()), c_void_p(squeeze_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    del convolution_90
    del primals_272
    del relu_87
    del squeeze_271
    del unsqueeze_590
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf112 = aten.convolution_backward(buf111, relu_86, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf111
    del primals_271
    buf113 = buf112[0]
    buf114 = buf112[1]
    del buf112
    buf115 = buf113; del buf113  # reuse
    buf116 = empty((512, ), device='cpu', dtype=torch.float32)
    buf117 = empty((512, ), device='cpu', dtype=torch.float32)
    buf118 = empty((512, ), device='cpu', dtype=torch.float32)
    buf119 = buf97; del buf97  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_16(c_void_p(buf115.data_ptr()), c_void_p(relu_86.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(relu_89.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(convolution_89.data_ptr()), c_void_p(unsqueeze_602.data_ptr()), c_void_p(squeeze_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()))
    del buf71
    del buf91
    del convolution_89
    del primals_269
    del relu_86
    del relu_89
    del squeeze_268
    del unsqueeze_602
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf120 = aten.convolution_backward(buf119, cat_11, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_11
    del primals_268
    buf121 = buf120[0]
    buf122 = buf120[1]
    del buf120
    buf123 = buf117; del buf117  # reuse
    buf124 = empty((512, ), device='cpu', dtype=torch.float32)
    buf125 = buf119; del buf119  # reuse
    buf126 = buf124; del buf124  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_17(c_void_p(buf126.data_ptr()), c_void_p(le_15.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(convolution_88.data_ptr()), c_void_p(unsqueeze_614.data_ptr()), c_void_p(squeeze_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()))
    del convolution_88
    del primals_266
    del squeeze_265
    del unsqueeze_614
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf127 = aten.convolution_backward(buf125, relu_84, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_265
    buf128 = buf127[0]
    buf129 = buf127[1]
    del buf127
    buf130 = buf109; del buf109  # reuse
    buf131 = empty((256, ), device='cpu', dtype=torch.float32)
    buf132 = empty((256, ), device='cpu', dtype=torch.float32)
    buf133 = buf128; del buf128  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(buf133.data_ptr()), c_void_p(relu_84.data_ptr()), c_void_p(convolution_87.data_ptr()), c_void_p(unsqueeze_626.data_ptr()), c_void_p(squeeze_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del convolution_87
    del primals_263
    del relu_84
    del squeeze_262
    del unsqueeze_626
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf134 = aten.convolution_backward(buf133, relu_83, primals_262, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf133
    del primals_262
    buf135 = buf134[0]
    buf136 = buf134[1]
    del buf134
    buf137 = buf131; del buf131  # reuse
    buf138 = empty((256, ), device='cpu', dtype=torch.float32)
    buf139 = empty((256, ), device='cpu', dtype=torch.float32)
    buf140 = buf135; del buf135  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(buf140.data_ptr()), c_void_p(relu_83.data_ptr()), c_void_p(convolution_86.data_ptr()), c_void_p(unsqueeze_638.data_ptr()), c_void_p(squeeze_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del convolution_86
    del primals_260
    del relu_83
    del squeeze_259
    del unsqueeze_638
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf141 = aten.convolution_backward(buf140, relu_82, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf140
    del primals_259
    buf142 = buf141[0]
    buf143 = buf141[1]
    del buf141
    buf144 = buf115; del buf115  # reuse
    buf145 = empty((512, ), device='cpu', dtype=torch.float32)
    buf146 = empty((512, ), device='cpu', dtype=torch.float32)
    buf147 = empty((512, ), device='cpu', dtype=torch.float32)
    buf148 = buf125; del buf125  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(buf144.data_ptr()), c_void_p(relu_82.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(le_15.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(convolution_85.data_ptr()), c_void_p(unsqueeze_650.data_ptr()), c_void_p(squeeze_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del buf121
    del buf142
    del convolution_85
    del le_15
    del primals_257
    del relu_82
    del squeeze_256
    del unsqueeze_650
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf149 = aten.convolution_backward(buf148, relu_81, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_256
    buf150 = buf149[0]
    buf151 = buf149[1]
    del buf149
    buf152 = buf138; del buf138  # reuse
    buf153 = empty((256, ), device='cpu', dtype=torch.float32)
    buf154 = empty((256, ), device='cpu', dtype=torch.float32)
    buf155 = buf150; del buf150  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21(c_void_p(buf155.data_ptr()), c_void_p(relu_81.data_ptr()), c_void_p(convolution_84.data_ptr()), c_void_p(unsqueeze_662.data_ptr()), c_void_p(squeeze_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del convolution_84
    del primals_254
    del relu_81
    del squeeze_253
    del unsqueeze_662
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf156 = aten.convolution_backward(buf155, relu_80, primals_253, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf155
    del primals_253
    buf157 = buf156[0]
    buf158 = buf156[1]
    del buf156
    buf159 = buf153; del buf153  # reuse
    buf160 = empty((256, ), device='cpu', dtype=torch.float32)
    buf161 = empty((256, ), device='cpu', dtype=torch.float32)
    buf162 = buf157; del buf157  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(buf162.data_ptr()), c_void_p(relu_80.data_ptr()), c_void_p(convolution_83.data_ptr()), c_void_p(unsqueeze_674.data_ptr()), c_void_p(squeeze_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()))
    del convolution_83
    del primals_251
    del relu_80
    del squeeze_250
    del unsqueeze_674
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf163 = aten.convolution_backward(buf162, relu_79, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf162
    del primals_250
    buf164 = buf163[0]
    buf165 = buf163[1]
    del buf163
    buf166 = buf146; del buf146  # reuse
    buf167 = empty((512, ), device='cpu', dtype=torch.float32)
    buf168 = buf148; del buf148  # reuse
    buf170 = buf168; del buf168  # reuse
    buf169 = buf167; del buf167  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(relu_79.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(convolution_82.data_ptr()), c_void_p(unsqueeze_686.data_ptr()), c_void_p(squeeze_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(buf166.data_ptr()))
    del convolution_82
    del primals_248
    del squeeze_247
    del unsqueeze_686
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf171 = aten.convolution_backward(buf170, cat_10, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_10
    del primals_247
    buf172 = buf171[0]
    buf173 = buf171[1]
    del buf171
    buf174 = buf144; del buf144  # reuse
    buf175 = empty((512, ), device='cpu', dtype=torch.float32)
    buf176 = empty((512, ), device='cpu', dtype=torch.float32)
    buf177 = empty((512, ), device='cpu', dtype=torch.float32)
    buf178 = buf170; del buf170  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf174.data_ptr()), c_void_p(le_22.data_ptr()), c_void_p(relu_79.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(convolution_81.data_ptr()), c_void_p(unsqueeze_698.data_ptr()), c_void_p(squeeze_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    del buf164
    del convolution_81
    del le_22
    del primals_245
    del relu_79
    del squeeze_244
    del unsqueeze_698
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf179 = aten.convolution_backward(buf178, relu_77, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_244
    buf180 = buf179[0]
    buf181 = buf179[1]
    del buf179
    buf182 = buf160; del buf160  # reuse
    buf183 = empty((256, ), device='cpu', dtype=torch.float32)
    buf184 = empty((256, ), device='cpu', dtype=torch.float32)
    buf185 = buf180; del buf180  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25(c_void_p(buf185.data_ptr()), c_void_p(relu_77.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(unsqueeze_710.data_ptr()), c_void_p(squeeze_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    del convolution_80
    del primals_242
    del relu_77
    del squeeze_241
    del unsqueeze_710
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf186 = aten.convolution_backward(buf185, relu_76, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf185
    del primals_241
    buf187 = buf186[0]
    buf188 = buf186[1]
    del buf186
    buf189 = buf183; del buf183  # reuse
    buf190 = empty((256, ), device='cpu', dtype=torch.float32)
    buf191 = empty((256, ), device='cpu', dtype=torch.float32)
    buf192 = buf187; del buf187  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26(c_void_p(buf192.data_ptr()), c_void_p(relu_76.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(unsqueeze_722.data_ptr()), c_void_p(squeeze_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    del convolution_79
    del primals_239
    del relu_76
    del squeeze_238
    del unsqueeze_722
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf193 = aten.convolution_backward(buf192, relu_75, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf192
    del primals_238
    buf194 = buf193[0]
    buf195 = buf193[1]
    del buf193
    buf196 = buf176; del buf176  # reuse
    buf197 = empty((512, ), device='cpu', dtype=torch.float32)
    buf198 = buf178; del buf178  # reuse
    buf200 = buf198; del buf198  # reuse
    buf199 = buf197; del buf197  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(buf200.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(relu_75.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(convolution_78.data_ptr()), c_void_p(unsqueeze_734.data_ptr()), c_void_p(squeeze_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(buf196.data_ptr()))
    del convolution_78
    del primals_236
    del squeeze_235
    del unsqueeze_734
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf201 = aten.convolution_backward(buf200, relu_74, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_235
    buf202 = buf201[0]
    buf203 = buf201[1]
    del buf201
    buf204 = buf190; del buf190  # reuse
    buf205 = empty((256, ), device='cpu', dtype=torch.float32)
    buf206 = empty((256, ), device='cpu', dtype=torch.float32)
    buf207 = buf202; del buf202  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28(c_void_p(buf207.data_ptr()), c_void_p(relu_74.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(unsqueeze_746.data_ptr()), c_void_p(squeeze_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del convolution_77
    del primals_233
    del relu_74
    del squeeze_232
    del unsqueeze_746
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf208 = aten.convolution_backward(buf207, relu_73, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf207
    del primals_232
    buf209 = buf208[0]
    buf210 = buf208[1]
    del buf208
    buf211 = buf205; del buf205  # reuse
    buf212 = empty((256, ), device='cpu', dtype=torch.float32)
    buf213 = empty((256, ), device='cpu', dtype=torch.float32)
    buf214 = buf209; del buf209  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29(c_void_p(buf214.data_ptr()), c_void_p(relu_73.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(unsqueeze_758.data_ptr()), c_void_p(squeeze_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del convolution_76
    del primals_230
    del relu_73
    del squeeze_229
    del unsqueeze_758
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf215 = aten.convolution_backward(buf214, relu_72, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf214
    del primals_229
    buf216 = buf215[0]
    buf217 = buf215[1]
    del buf215
    buf218 = buf174; del buf174  # reuse
    buf219 = empty((512, ), device='cpu', dtype=torch.float32)
    buf220 = empty((512, ), device='cpu', dtype=torch.float32)
    buf221 = empty((512, ), device='cpu', dtype=torch.float32)
    buf222 = buf200; del buf200  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_30(c_void_p(buf218.data_ptr()), c_void_p(relu_72.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(relu_75.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(unsqueeze_770.data_ptr()), c_void_p(squeeze_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    del buf172
    del buf194
    del buf216
    del convolution_75
    del primals_227
    del relu_72
    del relu_75
    del squeeze_226
    del unsqueeze_770
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf223 = aten.convolution_backward(buf222, cat_9, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_9
    del primals_226
    buf224 = buf223[0]
    buf225 = buf223[1]
    del buf223
    buf226 = buf220; del buf220  # reuse
    buf227 = empty((512, ), device='cpu', dtype=torch.float32)
    buf228 = buf222; del buf222  # reuse
    buf229 = buf227; del buf227  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_31(c_void_p(buf229.data_ptr()), c_void_p(le_29.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(unsqueeze_782.data_ptr()), c_void_p(squeeze_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()))
    del convolution_74
    del primals_224
    del squeeze_223
    del unsqueeze_782
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf230 = aten.convolution_backward(buf228, relu_70, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_223
    buf231 = buf230[0]
    buf232 = buf230[1]
    del buf230
    buf233 = buf212; del buf212  # reuse
    buf234 = empty((256, ), device='cpu', dtype=torch.float32)
    buf235 = empty((256, ), device='cpu', dtype=torch.float32)
    buf236 = buf231; del buf231  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(buf236.data_ptr()), c_void_p(relu_70.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(unsqueeze_794.data_ptr()), c_void_p(squeeze_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del convolution_73
    del primals_221
    del relu_70
    del squeeze_220
    del unsqueeze_794
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf237 = aten.convolution_backward(buf236, relu_69, primals_220, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf236
    del primals_220
    buf238 = buf237[0]
    buf239 = buf237[1]
    del buf237
    buf240 = buf234; del buf234  # reuse
    buf241 = empty((256, ), device='cpu', dtype=torch.float32)
    buf242 = empty((256, ), device='cpu', dtype=torch.float32)
    buf243 = buf238; del buf238  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33(c_void_p(buf243.data_ptr()), c_void_p(relu_69.data_ptr()), c_void_p(convolution_72.data_ptr()), c_void_p(unsqueeze_806.data_ptr()), c_void_p(squeeze_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    del convolution_72
    del primals_218
    del relu_69
    del squeeze_217
    del unsqueeze_806
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf244 = aten.convolution_backward(buf243, relu_68, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf243
    del primals_217
    buf245 = buf244[0]
    buf246 = buf244[1]
    del buf244
    buf247 = buf218; del buf218  # reuse
    buf248 = empty((512, ), device='cpu', dtype=torch.float32)
    buf249 = empty((512, ), device='cpu', dtype=torch.float32)
    buf250 = empty((512, ), device='cpu', dtype=torch.float32)
    buf251 = buf228; del buf228  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_34(c_void_p(buf247.data_ptr()), c_void_p(relu_68.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(le_29.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(unsqueeze_818.data_ptr()), c_void_p(squeeze_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    del buf224
    del buf245
    del convolution_71
    del le_29
    del primals_215
    del relu_68
    del squeeze_214
    del unsqueeze_818
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf252 = aten.convolution_backward(buf251, relu_67, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_214
    buf253 = buf252[0]
    buf254 = buf252[1]
    del buf252
    buf255 = buf241; del buf241  # reuse
    buf256 = empty((256, ), device='cpu', dtype=torch.float32)
    buf257 = empty((256, ), device='cpu', dtype=torch.float32)
    buf258 = buf253; del buf253  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35(c_void_p(buf258.data_ptr()), c_void_p(relu_67.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(unsqueeze_830.data_ptr()), c_void_p(squeeze_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()))
    del convolution_70
    del primals_212
    del relu_67
    del squeeze_211
    del unsqueeze_830
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf259 = aten.convolution_backward(buf258, relu_66, primals_211, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf258
    del primals_211
    buf260 = buf259[0]
    buf261 = buf259[1]
    del buf259
    buf262 = buf256; del buf256  # reuse
    buf263 = empty((256, ), device='cpu', dtype=torch.float32)
    buf264 = empty((256, ), device='cpu', dtype=torch.float32)
    buf265 = buf260; del buf260  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf265.data_ptr()), c_void_p(relu_66.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(unsqueeze_842.data_ptr()), c_void_p(squeeze_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    del convolution_69
    del primals_209
    del relu_66
    del squeeze_208
    del unsqueeze_842
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf266 = aten.convolution_backward(buf265, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf265
    del primals_208
    buf267 = buf266[0]
    buf268 = buf266[1]
    del buf266
    buf269 = buf249; del buf249  # reuse
    buf270 = empty((512, ), device='cpu', dtype=torch.float32)
    buf271 = buf251; del buf251  # reuse
    buf273 = buf271; del buf271  # reuse
    buf272 = buf270; del buf270  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_37(c_void_p(buf273.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(relu_65.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(unsqueeze_854.data_ptr()), c_void_p(squeeze_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf269.data_ptr()))
    del convolution_68
    del primals_206
    del squeeze_205
    del unsqueeze_854
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf274 = aten.convolution_backward(buf273, cat_8, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_8
    del primals_205
    buf275 = buf274[0]
    buf276 = buf274[1]
    del buf274
    buf277 = buf247; del buf247  # reuse
    buf278 = empty((512, ), device='cpu', dtype=torch.float32)
    buf279 = empty((512, ), device='cpu', dtype=torch.float32)
    buf280 = empty((512, ), device='cpu', dtype=torch.float32)
    buf281 = buf273; del buf273  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38(c_void_p(buf277.data_ptr()), c_void_p(le_36.data_ptr()), c_void_p(relu_65.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(unsqueeze_866.data_ptr()), c_void_p(squeeze_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del buf267
    del convolution_67
    del le_36
    del primals_203
    del relu_65
    del squeeze_202
    del unsqueeze_866
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf282 = aten.convolution_backward(buf281, relu_63, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_202
    buf283 = buf282[0]
    buf284 = buf282[1]
    del buf282
    buf285 = buf263; del buf263  # reuse
    buf286 = empty((256, ), device='cpu', dtype=torch.float32)
    buf287 = empty((256, ), device='cpu', dtype=torch.float32)
    buf288 = buf283; del buf283  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(buf288.data_ptr()), c_void_p(relu_63.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(unsqueeze_878.data_ptr()), c_void_p(squeeze_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    del convolution_66
    del primals_200
    del relu_63
    del squeeze_199
    del unsqueeze_878
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf289 = aten.convolution_backward(buf288, relu_62, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf288
    del primals_199
    buf290 = buf289[0]
    buf291 = buf289[1]
    del buf289
    buf292 = buf286; del buf286  # reuse
    buf293 = empty((256, ), device='cpu', dtype=torch.float32)
    buf294 = empty((256, ), device='cpu', dtype=torch.float32)
    buf295 = buf290; del buf290  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40(c_void_p(buf295.data_ptr()), c_void_p(relu_62.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_890.data_ptr()), c_void_p(squeeze_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    del convolution_65
    del primals_197
    del relu_62
    del squeeze_196
    del unsqueeze_890
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf296 = aten.convolution_backward(buf295, relu_61, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf295
    del primals_196
    buf297 = buf296[0]
    buf298 = buf296[1]
    del buf296
    buf299 = buf279; del buf279  # reuse
    buf300 = empty((512, ), device='cpu', dtype=torch.float32)
    buf301 = buf281; del buf281  # reuse
    buf303 = buf301; del buf301  # reuse
    buf302 = buf300; del buf300  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_41(c_void_p(buf303.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(relu_61.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_902.data_ptr()), c_void_p(squeeze_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(buf299.data_ptr()))
    del convolution_64
    del primals_194
    del squeeze_193
    del unsqueeze_902
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf304 = aten.convolution_backward(buf303, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_193
    buf305 = buf304[0]
    buf306 = buf304[1]
    del buf304
    buf307 = buf293; del buf293  # reuse
    buf308 = empty((256, ), device='cpu', dtype=torch.float32)
    buf309 = empty((256, ), device='cpu', dtype=torch.float32)
    buf310 = buf305; del buf305  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42(c_void_p(buf310.data_ptr()), c_void_p(relu_60.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(unsqueeze_914.data_ptr()), c_void_p(squeeze_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()))
    del convolution_63
    del primals_191
    del relu_60
    del squeeze_190
    del unsqueeze_914
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf311 = aten.convolution_backward(buf310, relu_59, primals_190, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf310
    del primals_190
    buf312 = buf311[0]
    buf313 = buf311[1]
    del buf311
    buf314 = buf308; del buf308  # reuse
    buf315 = empty((256, ), device='cpu', dtype=torch.float32)
    buf316 = empty((256, ), device='cpu', dtype=torch.float32)
    buf317 = buf312; del buf312  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(buf317.data_ptr()), c_void_p(relu_59.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(unsqueeze_926.data_ptr()), c_void_p(squeeze_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()))
    del convolution_62
    del primals_188
    del relu_59
    del squeeze_187
    del unsqueeze_926
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf318 = aten.convolution_backward(buf317, relu_58, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf317
    del primals_187
    buf319 = buf318[0]
    buf320 = buf318[1]
    del buf318
    buf321 = buf277; del buf277  # reuse
    buf322 = empty((512, ), device='cpu', dtype=torch.float32)
    buf323 = empty((512, ), device='cpu', dtype=torch.float32)
    buf324 = empty((512, ), device='cpu', dtype=torch.float32)
    buf325 = buf303; del buf303  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_44(c_void_p(buf321.data_ptr()), c_void_p(relu_58.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(relu_61.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_938.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del buf297
    del buf319
    del convolution_61
    del primals_185
    del relu_58
    del relu_61
    del squeeze_184
    del unsqueeze_938
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf326 = aten.convolution_backward(buf325, cat_7, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_7
    del primals_184
    buf327 = buf326[0]
    buf328 = buf326[1]
    del buf326
    buf329 = buf323; del buf323  # reuse
    buf330 = empty((512, ), device='cpu', dtype=torch.float32)
    buf331 = buf325; del buf325  # reuse
    buf332 = buf330; del buf330  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_45(c_void_p(buf332.data_ptr()), c_void_p(le_43.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_950.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()))
    del convolution_60
    del primals_182
    del squeeze_181
    del unsqueeze_950
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf333 = aten.convolution_backward(buf331, relu_56, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_181
    buf334 = buf333[0]
    buf335 = buf333[1]
    del buf333
    buf336 = buf315; del buf315  # reuse
    buf337 = empty((256, ), device='cpu', dtype=torch.float32)
    buf338 = empty((256, ), device='cpu', dtype=torch.float32)
    buf339 = buf334; del buf334  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_46(c_void_p(buf339.data_ptr()), c_void_p(relu_56.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_962.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()))
    del convolution_59
    del primals_179
    del relu_56
    del squeeze_178
    del unsqueeze_962
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf340 = aten.convolution_backward(buf339, relu_55, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf339
    del primals_178
    buf341 = buf340[0]
    buf342 = buf340[1]
    del buf340
    buf343 = buf337; del buf337  # reuse
    buf344 = empty((256, ), device='cpu', dtype=torch.float32)
    buf345 = empty((256, ), device='cpu', dtype=torch.float32)
    buf346 = buf341; del buf341  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47(c_void_p(buf346.data_ptr()), c_void_p(relu_55.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(unsqueeze_974.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()))
    del convolution_58
    del primals_176
    del relu_55
    del squeeze_175
    del unsqueeze_974
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf347 = aten.convolution_backward(buf346, relu_54, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf346
    del primals_175
    buf348 = buf347[0]
    buf349 = buf347[1]
    del buf347
    buf350 = buf321; del buf321  # reuse
    buf351 = empty((512, ), device='cpu', dtype=torch.float32)
    buf352 = empty((512, ), device='cpu', dtype=torch.float32)
    buf353 = empty((512, ), device='cpu', dtype=torch.float32)
    buf354 = buf331; del buf331  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_48(c_void_p(buf350.data_ptr()), c_void_p(relu_54.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(le_43.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(unsqueeze_986.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    del buf327
    del buf348
    del convolution_57
    del le_43
    del primals_173
    del relu_54
    del squeeze_172
    del unsqueeze_986
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf355 = aten.convolution_backward(buf354, relu_53, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_172
    buf356 = buf355[0]
    buf357 = buf355[1]
    del buf355
    buf358 = buf344; del buf344  # reuse
    buf359 = empty((256, ), device='cpu', dtype=torch.float32)
    buf360 = empty((256, ), device='cpu', dtype=torch.float32)
    buf361 = buf356; del buf356  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49(c_void_p(buf361.data_ptr()), c_void_p(relu_53.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_998.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()))
    del convolution_56
    del primals_170
    del relu_53
    del squeeze_169
    del unsqueeze_998
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf362 = aten.convolution_backward(buf361, relu_52, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf361
    del primals_169
    buf363 = buf362[0]
    buf364 = buf362[1]
    del buf362
    buf365 = buf359; del buf359  # reuse
    buf366 = empty((256, ), device='cpu', dtype=torch.float32)
    buf367 = empty((256, ), device='cpu', dtype=torch.float32)
    buf368 = buf363; del buf363  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50(c_void_p(buf368.data_ptr()), c_void_p(relu_52.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_1010.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()))
    del convolution_55
    del primals_167
    del relu_52
    del squeeze_166
    del unsqueeze_1010
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf369 = aten.convolution_backward(buf368, relu_51, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf368
    del primals_166
    buf370 = buf369[0]
    buf371 = buf369[1]
    del buf369
    buf372 = buf352; del buf352  # reuse
    buf373 = empty((512, ), device='cpu', dtype=torch.float32)
    buf374 = buf354; del buf354  # reuse
    buf376 = buf374; del buf374  # reuse
    buf375 = buf373; del buf373  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_51(c_void_p(buf376.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(relu_51.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_1022.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf372.data_ptr()))
    del convolution_54
    del primals_164
    del squeeze_163
    del unsqueeze_1022
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf377 = aten.convolution_backward(buf376, cat_6, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_6
    del primals_163
    buf378 = buf377[0]
    buf379 = buf377[1]
    del buf377
    buf380 = buf350; del buf350  # reuse
    buf381 = empty((512, ), device='cpu', dtype=torch.float32)
    buf382 = empty((512, ), device='cpu', dtype=torch.float32)
    buf383 = empty((512, ), device='cpu', dtype=torch.float32)
    buf384 = buf376; del buf376  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_52(c_void_p(buf380.data_ptr()), c_void_p(le_50.data_ptr()), c_void_p(relu_51.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_1034.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()))
    del buf275
    del buf370
    del convolution_53
    del le_50
    del primals_161
    del relu_51
    del squeeze_160
    del unsqueeze_1034
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf385 = aten.convolution_backward(buf384, relu_49, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_160
    buf386 = buf385[0]
    buf387 = buf385[1]
    del buf385
    buf388 = buf366; del buf366  # reuse
    buf389 = empty((256, ), device='cpu', dtype=torch.float32)
    buf390 = empty((256, ), device='cpu', dtype=torch.float32)
    buf391 = buf386; del buf386  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53(c_void_p(buf391.data_ptr()), c_void_p(relu_49.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_1046.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()))
    del convolution_52
    del primals_158
    del relu_49
    del squeeze_157
    del unsqueeze_1046
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf392 = aten.convolution_backward(buf391, relu_48, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf391
    del primals_157
    buf393 = buf392[0]
    buf394 = buf392[1]
    del buf392
    buf395 = buf389; del buf389  # reuse
    buf396 = empty((256, ), device='cpu', dtype=torch.float32)
    buf397 = empty((256, ), device='cpu', dtype=torch.float32)
    buf398 = buf393; del buf393  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54(c_void_p(buf398.data_ptr()), c_void_p(relu_48.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_1058.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()))
    del convolution_51
    del primals_155
    del relu_48
    del squeeze_154
    del unsqueeze_1058
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf399 = aten.convolution_backward(buf398, relu_47, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf398
    del primals_154
    buf400 = buf399[0]
    buf401 = buf399[1]
    del buf399
    buf402 = buf382; del buf382  # reuse
    buf403 = empty((512, ), device='cpu', dtype=torch.float32)
    buf404 = buf384; del buf384  # reuse
    buf406 = buf404; del buf404  # reuse
    buf405 = buf403; del buf403  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_55(c_void_p(buf406.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(relu_47.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_1070.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf402.data_ptr()))
    del convolution_50
    del primals_152
    del squeeze_151
    del unsqueeze_1070
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf407 = aten.convolution_backward(buf406, relu_46, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_151
    buf408 = buf407[0]
    buf409 = buf407[1]
    del buf407
    buf410 = buf396; del buf396  # reuse
    buf411 = empty((256, ), device='cpu', dtype=torch.float32)
    buf412 = empty((256, ), device='cpu', dtype=torch.float32)
    buf413 = buf408; del buf408  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56(c_void_p(buf413.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_1082.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()))
    del convolution_49
    del primals_149
    del relu_46
    del squeeze_148
    del unsqueeze_1082
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf414 = aten.convolution_backward(buf413, relu_45, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf413
    del primals_148
    buf415 = buf414[0]
    buf416 = buf414[1]
    del buf414
    buf417 = buf411; del buf411  # reuse
    buf418 = empty((256, ), device='cpu', dtype=torch.float32)
    buf419 = empty((256, ), device='cpu', dtype=torch.float32)
    buf420 = buf415; del buf415  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_57(c_void_p(buf420.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_1094.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()))
    del convolution_48
    del primals_146
    del relu_45
    del squeeze_145
    del unsqueeze_1094
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf421 = aten.convolution_backward(buf420, relu_44, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf420
    del primals_145
    buf422 = buf421[0]
    buf423 = buf421[1]
    del buf421
    buf424 = buf380; del buf380  # reuse
    buf425 = empty((512, ), device='cpu', dtype=torch.float32)
    buf426 = empty((512, ), device='cpu', dtype=torch.float32)
    buf427 = empty((512, ), device='cpu', dtype=torch.float32)
    buf428 = buf406; del buf406  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_58(c_void_p(buf424.data_ptr()), c_void_p(relu_44.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(relu_47.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_1106.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()))
    del buf378
    del buf400
    del convolution_47
    del primals_143
    del relu_44
    del relu_47
    del squeeze_142
    del unsqueeze_1106
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf429 = aten.convolution_backward(buf428, cat_5, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_5
    del primals_142
    buf430 = buf429[0]
    buf431 = buf429[1]
    del buf429
    buf432 = buf426; del buf426  # reuse
    buf433 = empty((512, ), device='cpu', dtype=torch.float32)
    buf434 = buf428; del buf428  # reuse
    buf435 = buf433; del buf433  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_59(c_void_p(buf435.data_ptr()), c_void_p(le_57.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_1118.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf434.data_ptr()))
    del convolution_46
    del primals_140
    del squeeze_139
    del unsqueeze_1118
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf436 = aten.convolution_backward(buf434, relu_42, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_139
    buf437 = buf436[0]
    buf438 = buf436[1]
    del buf436
    buf439 = buf418; del buf418  # reuse
    buf440 = empty((256, ), device='cpu', dtype=torch.float32)
    buf441 = empty((256, ), device='cpu', dtype=torch.float32)
    buf442 = buf437; del buf437  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60(c_void_p(buf442.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_1130.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()))
    del convolution_45
    del primals_137
    del relu_42
    del squeeze_136
    del unsqueeze_1130
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf443 = aten.convolution_backward(buf442, relu_41, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf442
    del primals_136
    buf444 = buf443[0]
    buf445 = buf443[1]
    del buf443
    buf446 = buf440; del buf440  # reuse
    buf447 = empty((256, ), device='cpu', dtype=torch.float32)
    buf448 = empty((256, ), device='cpu', dtype=torch.float32)
    buf449 = buf444; del buf444  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61(c_void_p(buf449.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_1142.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()))
    del convolution_44
    del primals_134
    del relu_41
    del squeeze_133
    del unsqueeze_1142
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf450 = aten.convolution_backward(buf449, relu_40, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf449
    del primals_133
    buf451 = buf450[0]
    buf452 = buf450[1]
    del buf450
    buf453 = buf424; del buf424  # reuse
    buf454 = empty((512, ), device='cpu', dtype=torch.float32)
    buf455 = empty((512, ), device='cpu', dtype=torch.float32)
    buf475 = empty((512, ), device='cpu', dtype=torch.float32)
    buf456 = empty((512, ), device='cpu', dtype=torch.float32)
    buf457 = buf434; del buf434  # reuse
    buf477 = buf422; del buf422  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_62(c_void_p(buf453.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(le_57.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_1154.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_1190.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf477.data_ptr()))
    del buf430
    del buf451
    del buf453
    del buf455
    del convolution_40
    del convolution_43
    del le_57
    del primals_122
    del primals_131
    del relu_40
    del squeeze_130
    del unsqueeze_1154
    del unsqueeze_1190
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf458 = aten.convolution_backward(buf457, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf457
    del primals_130
    buf459 = buf458[0]
    buf460 = buf458[1]
    del buf458
    buf461 = buf447; del buf447  # reuse
    buf462 = empty((256, ), device='cpu', dtype=torch.float32)
    buf463 = empty((256, ), device='cpu', dtype=torch.float32)
    buf464 = buf459; del buf459  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_63(c_void_p(buf464.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_1166.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()))
    del convolution_42
    del primals_128
    del relu_39
    del squeeze_127
    del unsqueeze_1166
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf465 = aten.convolution_backward(buf464, relu_38, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf464
    del primals_127
    buf466 = buf465[0]
    buf467 = buf465[1]
    del buf465
    buf468 = buf462; del buf462  # reuse
    buf469 = empty((256, ), device='cpu', dtype=torch.float32)
    buf470 = empty((256, ), device='cpu', dtype=torch.float32)
    buf471 = buf466; del buf466  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64(c_void_p(buf471.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_1178.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()))
    del convolution_41
    del primals_125
    del relu_38
    del squeeze_124
    del unsqueeze_1178
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf472 = aten.convolution_backward(buf471, relu_37, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_124
    buf473 = buf472[0]
    buf474 = buf472[1]
    del buf472
    buf476 = buf475; del buf475  # reuse
    cpp_fused_native_batch_norm_backward_65(c_void_p(buf476.data_ptr()), c_void_p(squeeze_121.data_ptr()))
    del squeeze_121
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf478 = aten.convolution_backward(buf477, getitem_88, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf477
    del getitem_88
    del primals_121
    buf479 = buf478[0]
    buf480 = buf478[1]
    del buf478
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf481 = aten.max_pool2d_with_indices_backward(buf479, relu_37, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_89)
    del buf479
    buf482 = buf481
    del buf481
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf483 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf69, (8, 256, 14, 14), (551936, 1, 39424, 2816), 1024), relu_37, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_89)
    del buf69
    del getitem_89
    buf484 = buf483
    del buf483
    buf485 = buf469; del buf469  # reuse
    buf486 = empty((256, ), device='cpu', dtype=torch.float32)
    buf487 = buf471; del buf471  # reuse
    buf489 = buf487; del buf487  # reuse
    buf488 = buf486; del buf486  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_66(c_void_p(buf489.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_1202.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf485.data_ptr()))
    del convolution_39
    del primals_119
    del squeeze_118
    del unsqueeze_1202
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf490 = aten.convolution_backward(buf489, cat_4, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_4
    del primals_118
    buf491 = buf490[0]
    buf492 = buf490[1]
    del buf490
    buf493 = buf473; del buf473  # reuse
    buf494 = empty((256, ), device='cpu', dtype=torch.float32)
    buf495 = empty((256, ), device='cpu', dtype=torch.float32)
    buf496 = empty((256, ), device='cpu', dtype=torch.float32)
    buf497 = buf489; del buf489  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_67(c_void_p(buf493.data_ptr()), c_void_p(le_64.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_1214.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()))
    del buf482
    del buf484
    del convolution_38
    del le_64
    del primals_116
    del relu_37
    del squeeze_115
    del unsqueeze_1214
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf498 = aten.convolution_backward(buf497, relu_35, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_115
    buf499 = buf498[0]
    buf500 = buf498[1]
    del buf498
    buf501 = empty((128, ), device='cpu', dtype=torch.float32)
    buf502 = empty((128, ), device='cpu', dtype=torch.float32)
    buf503 = empty((128, ), device='cpu', dtype=torch.float32)
    buf504 = buf499; del buf499  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68(c_void_p(buf504.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_1226.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()))
    del convolution_37
    del primals_113
    del relu_35
    del squeeze_112
    del unsqueeze_1226
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf505 = aten.convolution_backward(buf504, relu_34, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf504
    del primals_112
    buf506 = buf505[0]
    buf507 = buf505[1]
    del buf505
    buf508 = buf502; del buf502  # reuse
    buf509 = empty((128, ), device='cpu', dtype=torch.float32)
    buf510 = empty((128, ), device='cpu', dtype=torch.float32)
    buf511 = buf506; del buf506  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69(c_void_p(buf511.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_1238.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()))
    del convolution_36
    del primals_110
    del relu_34
    del squeeze_109
    del unsqueeze_1238
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf512 = aten.convolution_backward(buf511, relu_33, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf511
    del primals_109
    buf513 = buf512[0]
    buf514 = buf512[1]
    del buf512
    buf515 = buf495; del buf495  # reuse
    buf516 = empty((256, ), device='cpu', dtype=torch.float32)
    buf517 = buf497; del buf497  # reuse
    buf519 = buf517; del buf517  # reuse
    buf518 = buf516; del buf516  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_70(c_void_p(buf519.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_1250.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf515.data_ptr()))
    del convolution_35
    del primals_107
    del squeeze_106
    del unsqueeze_1250
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf520 = aten.convolution_backward(buf519, relu_32, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_106
    buf521 = buf520[0]
    buf522 = buf520[1]
    del buf520
    buf523 = buf509; del buf509  # reuse
    buf524 = empty((128, ), device='cpu', dtype=torch.float32)
    buf525 = empty((128, ), device='cpu', dtype=torch.float32)
    buf526 = buf521; del buf521  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_71(c_void_p(buf526.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_1262.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf525.data_ptr()))
    del convolution_34
    del primals_104
    del relu_32
    del squeeze_103
    del unsqueeze_1262
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf527 = aten.convolution_backward(buf526, relu_31, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf526
    del primals_103
    buf528 = buf527[0]
    buf529 = buf527[1]
    del buf527
    buf530 = buf524; del buf524  # reuse
    buf531 = empty((128, ), device='cpu', dtype=torch.float32)
    buf532 = empty((128, ), device='cpu', dtype=torch.float32)
    buf533 = buf528; del buf528  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_72(c_void_p(buf533.data_ptr()), c_void_p(relu_31.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_1274.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    del convolution_33
    del primals_101
    del relu_31
    del squeeze_100
    del unsqueeze_1274
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf534 = aten.convolution_backward(buf533, relu_30, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf533
    del primals_100
    buf535 = buf534[0]
    buf536 = buf534[1]
    del buf534
    buf537 = buf493; del buf493  # reuse
    buf538 = empty((256, ), device='cpu', dtype=torch.float32)
    buf539 = empty((256, ), device='cpu', dtype=torch.float32)
    buf540 = empty((256, ), device='cpu', dtype=torch.float32)
    buf541 = buf519; del buf519  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_73(c_void_p(buf537.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_1286.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()))
    del buf513
    del buf535
    del convolution_32
    del primals_98
    del relu_30
    del relu_33
    del squeeze_97
    del unsqueeze_1286
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf542 = aten.convolution_backward(buf541, cat_3, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_3
    del primals_97
    buf543 = buf542[0]
    buf544 = buf542[1]
    del buf542
    buf545 = buf539; del buf539  # reuse
    buf546 = empty((256, ), device='cpu', dtype=torch.float32)
    buf547 = buf541; del buf541  # reuse
    buf548 = buf546; del buf546  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_74(c_void_p(buf548.data_ptr()), c_void_p(le_71.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_1298.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf547.data_ptr()))
    del convolution_31
    del primals_95
    del squeeze_94
    del unsqueeze_1298
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf549 = aten.convolution_backward(buf547, relu_28, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_94
    buf550 = buf549[0]
    buf551 = buf549[1]
    del buf549
    buf552 = buf531; del buf531  # reuse
    buf553 = empty((128, ), device='cpu', dtype=torch.float32)
    buf554 = empty((128, ), device='cpu', dtype=torch.float32)
    buf555 = buf550; del buf550  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75(c_void_p(buf555.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_1310.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()))
    del convolution_30
    del primals_92
    del relu_28
    del squeeze_91
    del unsqueeze_1310
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf556 = aten.convolution_backward(buf555, relu_27, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf555
    del primals_91
    buf557 = buf556[0]
    buf558 = buf556[1]
    del buf556
    buf559 = buf553; del buf553  # reuse
    buf560 = empty((128, ), device='cpu', dtype=torch.float32)
    buf561 = empty((128, ), device='cpu', dtype=torch.float32)
    buf562 = buf557; del buf557  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76(c_void_p(buf562.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_1322.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()))
    del convolution_29
    del primals_89
    del relu_27
    del squeeze_88
    del unsqueeze_1322
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf563 = aten.convolution_backward(buf562, relu_26, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf562
    del primals_88
    buf564 = buf563[0]
    buf565 = buf563[1]
    del buf563
    buf566 = buf537; del buf537  # reuse
    buf567 = empty((256, ), device='cpu', dtype=torch.float32)
    buf568 = empty((256, ), device='cpu', dtype=torch.float32)
    buf569 = empty((256, ), device='cpu', dtype=torch.float32)
    buf570 = buf547; del buf547  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_77(c_void_p(buf566.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(le_71.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_1334.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()))
    del buf543
    del buf564
    del convolution_28
    del le_71
    del primals_86
    del relu_26
    del squeeze_85
    del unsqueeze_1334
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf571 = aten.convolution_backward(buf570, relu_25, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_85
    buf572 = buf571[0]
    buf573 = buf571[1]
    del buf571
    buf574 = buf560; del buf560  # reuse
    buf575 = empty((128, ), device='cpu', dtype=torch.float32)
    buf576 = empty((128, ), device='cpu', dtype=torch.float32)
    buf577 = buf572; del buf572  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78(c_void_p(buf577.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_1346.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()))
    del convolution_27
    del primals_83
    del relu_25
    del squeeze_82
    del unsqueeze_1346
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf578 = aten.convolution_backward(buf577, relu_24, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf577
    del primals_82
    buf579 = buf578[0]
    buf580 = buf578[1]
    del buf578
    buf581 = buf575; del buf575  # reuse
    buf582 = empty((128, ), device='cpu', dtype=torch.float32)
    buf583 = empty((128, ), device='cpu', dtype=torch.float32)
    buf584 = buf579; del buf579  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79(c_void_p(buf584.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_1358.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf583.data_ptr()))
    del convolution_26
    del primals_80
    del relu_24
    del squeeze_79
    del unsqueeze_1358
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf585 = aten.convolution_backward(buf584, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf584
    del primals_79
    buf586 = buf585[0]
    buf587 = buf585[1]
    del buf585
    buf588 = buf568; del buf568  # reuse
    buf589 = empty((256, ), device='cpu', dtype=torch.float32)
    buf590 = buf570; del buf570  # reuse
    buf592 = buf590; del buf590  # reuse
    buf591 = buf589; del buf589  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_80(c_void_p(buf592.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_1370.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf588.data_ptr()))
    del convolution_25
    del primals_77
    del squeeze_76
    del unsqueeze_1370
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf593 = aten.convolution_backward(buf592, cat_2, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_2
    del primals_76
    buf594 = buf593[0]
    buf595 = buf593[1]
    del buf593
    buf596 = buf566; del buf566  # reuse
    buf597 = empty((256, ), device='cpu', dtype=torch.float32)
    buf598 = empty((256, ), device='cpu', dtype=torch.float32)
    buf599 = empty((256, ), device='cpu', dtype=torch.float32)
    buf600 = buf592; del buf592  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_81(c_void_p(buf596.data_ptr()), c_void_p(le_78.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_1382.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()))
    del buf586
    del convolution_24
    del le_78
    del primals_74
    del relu_23
    del squeeze_73
    del unsqueeze_1382
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf601 = aten.convolution_backward(buf600, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_73
    buf602 = buf601[0]
    buf603 = buf601[1]
    del buf601
    buf604 = buf582; del buf582  # reuse
    buf605 = empty((128, ), device='cpu', dtype=torch.float32)
    buf606 = empty((128, ), device='cpu', dtype=torch.float32)
    buf607 = buf602; del buf602  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82(c_void_p(buf607.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_1394.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()))
    del convolution_23
    del primals_71
    del relu_21
    del squeeze_70
    del unsqueeze_1394
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf608 = aten.convolution_backward(buf607, relu_20, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf607
    del primals_70
    buf609 = buf608[0]
    buf610 = buf608[1]
    del buf608
    buf611 = buf605; del buf605  # reuse
    buf612 = empty((128, ), device='cpu', dtype=torch.float32)
    buf613 = empty((128, ), device='cpu', dtype=torch.float32)
    buf614 = buf609; del buf609  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83(c_void_p(buf614.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_1406.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()))
    del convolution_22
    del primals_68
    del relu_20
    del squeeze_67
    del unsqueeze_1406
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf615 = aten.convolution_backward(buf614, relu_19, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf614
    del primals_67
    buf616 = buf615[0]
    buf617 = buf615[1]
    del buf615
    buf618 = buf598; del buf598  # reuse
    buf619 = empty((256, ), device='cpu', dtype=torch.float32)
    buf620 = buf600; del buf600  # reuse
    buf622 = buf620; del buf620  # reuse
    buf621 = buf619; del buf619  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_84(c_void_p(buf622.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_1418.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf618.data_ptr()))
    del convolution_21
    del primals_65
    del squeeze_64
    del unsqueeze_1418
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf623 = aten.convolution_backward(buf622, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_64
    buf624 = buf623[0]
    buf625 = buf623[1]
    del buf623
    buf626 = buf612; del buf612  # reuse
    buf627 = empty((128, ), device='cpu', dtype=torch.float32)
    buf628 = empty((128, ), device='cpu', dtype=torch.float32)
    buf629 = buf624; del buf624  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_85(c_void_p(buf629.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_1430.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf628.data_ptr()))
    del convolution_20
    del primals_62
    del relu_18
    del squeeze_61
    del unsqueeze_1430
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf630 = aten.convolution_backward(buf629, relu_17, primals_61, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf629
    del primals_61
    buf631 = buf630[0]
    buf632 = buf630[1]
    del buf630
    buf633 = buf627; del buf627  # reuse
    buf634 = empty((128, ), device='cpu', dtype=torch.float32)
    buf635 = empty((128, ), device='cpu', dtype=torch.float32)
    buf636 = buf631; del buf631  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_86(c_void_p(buf636.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_1442.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf635.data_ptr()))
    del convolution_19
    del primals_59
    del relu_17
    del squeeze_58
    del unsqueeze_1442
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf637 = aten.convolution_backward(buf636, relu_16, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf636
    del primals_58
    buf638 = buf637[0]
    buf639 = buf637[1]
    del buf637
    buf640 = buf596; del buf596  # reuse
    buf641 = empty((256, ), device='cpu', dtype=torch.float32)
    buf642 = empty((256, ), device='cpu', dtype=torch.float32)
    buf643 = empty((256, ), device='cpu', dtype=torch.float32)
    buf644 = buf622; del buf622  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_87(c_void_p(buf640.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_1454.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf644.data_ptr()))
    del buf594
    del buf616
    del convolution_18
    del primals_56
    del relu_16
    del relu_19
    del squeeze_55
    del unsqueeze_1454
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf645 = aten.convolution_backward(buf644, cat_1, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_1
    del primals_55
    buf646 = buf645[0]
    buf647 = buf645[1]
    del buf645
    buf648 = buf642; del buf642  # reuse
    buf649 = empty((256, ), device='cpu', dtype=torch.float32)
    buf650 = buf644; del buf644  # reuse
    buf651 = buf649; del buf649  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_88(c_void_p(buf651.data_ptr()), c_void_p(le_85.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_1466.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf650.data_ptr()))
    del convolution_17
    del primals_53
    del squeeze_52
    del unsqueeze_1466
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf652 = aten.convolution_backward(buf650, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_52
    buf653 = buf652[0]
    buf654 = buf652[1]
    del buf652
    buf655 = buf634; del buf634  # reuse
    buf656 = empty((128, ), device='cpu', dtype=torch.float32)
    buf657 = empty((128, ), device='cpu', dtype=torch.float32)
    buf658 = buf653; del buf653  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_89(c_void_p(buf658.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_1478.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf657.data_ptr()))
    del convolution_16
    del primals_50
    del relu_14
    del squeeze_49
    del unsqueeze_1478
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf659 = aten.convolution_backward(buf658, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf658
    del primals_49
    buf660 = buf659[0]
    buf661 = buf659[1]
    del buf659
    buf662 = buf656; del buf656  # reuse
    buf663 = empty((128, ), device='cpu', dtype=torch.float32)
    buf664 = empty((128, ), device='cpu', dtype=torch.float32)
    buf665 = buf660; del buf660  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90(c_void_p(buf665.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_1490.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf664.data_ptr()))
    del convolution_15
    del primals_47
    del relu_13
    del squeeze_46
    del unsqueeze_1490
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf666 = aten.convolution_backward(buf665, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf665
    del primals_46
    buf667 = buf666[0]
    buf668 = buf666[1]
    del buf666
    buf669 = buf640; del buf640  # reuse
    buf670 = empty((256, ), device='cpu', dtype=torch.float32)
    buf671 = empty((256, ), device='cpu', dtype=torch.float32)
    buf691 = empty((256, ), device='cpu', dtype=torch.float32)
    buf672 = empty((256, ), device='cpu', dtype=torch.float32)
    buf673 = buf650; del buf650  # reuse
    buf693 = buf638; del buf638  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_91(c_void_p(buf669.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(le_85.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_1502.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_1538.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf693.data_ptr()))
    del buf646
    del buf667
    del buf669
    del buf671
    del convolution_11
    del convolution_14
    del le_85
    del primals_35
    del primals_44
    del relu_12
    del squeeze_43
    del unsqueeze_1502
    del unsqueeze_1538
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf674 = aten.convolution_backward(buf673, relu_11, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf673
    del primals_43
    buf675 = buf674[0]
    buf676 = buf674[1]
    del buf674
    buf677 = buf663; del buf663  # reuse
    buf678 = empty((128, ), device='cpu', dtype=torch.float32)
    buf679 = empty((128, ), device='cpu', dtype=torch.float32)
    buf680 = buf675; del buf675  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92(c_void_p(buf680.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_1514.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf679.data_ptr()))
    del convolution_13
    del primals_41
    del relu_11
    del squeeze_40
    del unsqueeze_1514
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf681 = aten.convolution_backward(buf680, relu_10, primals_40, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf680
    del primals_40
    buf682 = buf681[0]
    buf683 = buf681[1]
    del buf681
    buf684 = buf678; del buf678  # reuse
    buf685 = empty((128, ), device='cpu', dtype=torch.float32)
    buf686 = empty((128, ), device='cpu', dtype=torch.float32)
    buf687 = buf682; del buf682  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93(c_void_p(buf687.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_1526.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf686.data_ptr()))
    del convolution_12
    del primals_38
    del relu_10
    del squeeze_37
    del unsqueeze_1526
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf688 = aten.convolution_backward(buf687, relu_9, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_37
    buf689 = buf688[0]
    buf690 = buf688[1]
    del buf688
    buf692 = buf691; del buf691  # reuse
    cpp_fused_native_batch_norm_backward_94(c_void_p(buf692.data_ptr()), c_void_p(squeeze_34.data_ptr()))
    del squeeze_34
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf694 = aten.convolution_backward(buf693, getitem_24, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf693
    del getitem_24
    del primals_34
    buf695 = buf694[0]
    buf696 = buf694[1]
    del buf694
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf697 = aten.max_pool2d_with_indices_backward(buf695, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_25)
    del buf695
    buf698 = buf697
    del buf697
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf699 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf491, (8, 128, 28, 28), (903168, 1, 32256, 1152), 512), relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_25)
    del buf491
    del getitem_25
    buf700 = buf699
    del buf699
    buf701 = buf685; del buf685  # reuse
    buf702 = empty((128, ), device='cpu', dtype=torch.float32)
    buf703 = buf687; del buf687  # reuse
    buf705 = buf703; del buf703  # reuse
    buf704 = buf702; del buf702  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_95(c_void_p(buf705.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_1550.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf701.data_ptr()))
    del convolution_10
    del primals_32
    del squeeze_31
    del unsqueeze_1550
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf706 = aten.convolution_backward(buf705, cat, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat
    del primals_31
    buf707 = buf706[0]
    buf708 = buf706[1]
    del buf706
    buf709 = buf689; del buf689  # reuse
    buf710 = empty((128, ), device='cpu', dtype=torch.float32)
    buf711 = empty((128, ), device='cpu', dtype=torch.float32)
    buf712 = empty((128, ), device='cpu', dtype=torch.float32)
    buf713 = buf705; del buf705  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_96(c_void_p(buf709.data_ptr()), c_void_p(le_92.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_1562.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf713.data_ptr()))
    del buf698
    del convolution_9
    del le_92
    del primals_29
    del relu_9
    del squeeze_28
    del unsqueeze_1562
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf714 = aten.convolution_backward(buf713, relu_7, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_28
    buf715 = buf714[0]
    buf716 = buf714[1]
    del buf714
    buf717 = empty((64, ), device='cpu', dtype=torch.float32)
    buf718 = empty((64, ), device='cpu', dtype=torch.float32)
    buf719 = empty((64, ), device='cpu', dtype=torch.float32)
    buf720 = buf715; del buf715  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_97(c_void_p(buf720.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_1574.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf719.data_ptr()))
    del convolution_8
    del primals_26
    del relu_7
    del squeeze_25
    del unsqueeze_1574
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf721 = aten.convolution_backward(buf720, relu_6, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf720
    del primals_25
    buf722 = buf721[0]
    buf723 = buf721[1]
    del buf721
    buf724 = buf718; del buf718  # reuse
    buf725 = empty((64, ), device='cpu', dtype=torch.float32)
    buf726 = empty((64, ), device='cpu', dtype=torch.float32)
    buf727 = buf722; del buf722  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_98(c_void_p(buf727.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_1586.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf724.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf726.data_ptr()))
    del convolution_7
    del primals_23
    del relu_6
    del squeeze_22
    del unsqueeze_1586
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf728 = aten.convolution_backward(buf727, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf727
    del primals_22
    buf729 = buf728[0]
    buf730 = buf728[1]
    del buf728
    buf731 = buf711; del buf711  # reuse
    buf732 = empty((128, ), device='cpu', dtype=torch.float32)
    buf753 = empty((128, ), device='cpu', dtype=torch.float32)
    buf733 = buf713; del buf713  # reuse
    buf754 = buf700; del buf700  # reuse
    buf735 = buf733; del buf733  # reuse
    buf756 = buf754; del buf754  # reuse
    buf734 = buf732; del buf732  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_99(c_void_p(buf735.data_ptr()), c_void_p(buf756.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_1598.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_1634.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf753.data_ptr()))
    del buf707
    del buf709
    del buf729
    del convolution_3
    del convolution_6
    del primals_11
    del primals_20
    del relu_5
    del squeeze_19
    del unsqueeze_1598
    del unsqueeze_1634
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf736 = aten.convolution_backward(buf735, relu_4, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf735
    del primals_19
    buf737 = buf736[0]
    buf738 = buf736[1]
    del buf736
    buf739 = buf725; del buf725  # reuse
    buf740 = empty((64, ), device='cpu', dtype=torch.float32)
    buf741 = empty((64, ), device='cpu', dtype=torch.float32)
    buf742 = buf737; del buf737  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_100(c_void_p(buf742.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_1610.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf741.data_ptr()))
    del convolution_5
    del primals_17
    del relu_4
    del squeeze_16
    del unsqueeze_1610
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf743 = aten.convolution_backward(buf742, relu_3, primals_16, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf742
    del primals_16
    buf744 = buf743[0]
    buf745 = buf743[1]
    del buf743
    buf746 = buf740; del buf740  # reuse
    buf747 = empty((64, ), device='cpu', dtype=torch.float32)
    buf748 = empty((64, ), device='cpu', dtype=torch.float32)
    buf749 = buf744; del buf744  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_101(c_void_p(buf749.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_1622.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf748.data_ptr()))
    del buf747
    del convolution_4
    del primals_14
    del relu_3
    del squeeze_13
    del unsqueeze_1622
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf750 = aten.convolution_backward(buf749, relu_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf749
    del primals_13
    buf751 = buf750[0]
    buf752 = buf750[1]
    del buf750
    buf755 = buf753; del buf753  # reuse
    cpp_fused_native_batch_norm_backward_102(c_void_p(buf755.data_ptr()), c_void_p(squeeze_10.data_ptr()))
    del squeeze_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf757 = aten.convolution_backward(buf756, getitem_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf756
    del getitem_6
    del primals_10
    buf758 = buf757[0]
    buf759 = buf757[1]
    del buf757
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf760 = aten.max_pool2d_with_indices_backward(buf758, relu_2, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_7)
    del buf758
    del getitem_7
    buf761 = buf760
    del buf760
    buf762 = empty((32, ), device='cpu', dtype=torch.float32)
    buf763 = empty((32, ), device='cpu', dtype=torch.float32)
    buf764 = buf751; del buf751  # reuse
    buf765 = buf763; del buf763  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_103(c_void_p(buf764.data_ptr()), c_void_p(buf765.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_1646.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf762.data_ptr()))
    del buf761
    del convolution_2
    del primals_8
    del relu_2
    del squeeze_7
    del unsqueeze_1646
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf766 = aten.convolution_backward(buf764, relu_1, primals_7, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf764
    del primals_7
    buf767 = buf766[0]
    buf768 = buf766[1]
    del buf766
    buf769 = empty((16, ), device='cpu', dtype=torch.float32)
    buf770 = empty((16, ), device='cpu', dtype=torch.float32)
    buf771 = empty((16, ), device='cpu', dtype=torch.float32)
    buf772 = buf767; del buf767  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_104(c_void_p(buf772.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_1658.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf771.data_ptr()))
    del convolution_1
    del primals_5
    del relu_1
    del squeeze_4
    del unsqueeze_1658
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf773 = aten.convolution_backward(buf772, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf772
    del primals_4
    buf774 = buf773[0]
    buf775 = buf773[1]
    del buf773
    buf776 = buf770; del buf770  # reuse
    buf777 = empty((16, ), device='cpu', dtype=torch.float32)
    buf778 = empty((16, ), device='cpu', dtype=torch.float32)
    buf779 = buf774; del buf774  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_105(c_void_p(buf779.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_1670.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(buf778.data_ptr()))
    del buf777
    del convolution
    del primals_2
    del relu
    del squeeze_1
    del unsqueeze_1670
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf780 = aten.convolution_backward(buf779, primals_633, primals_1, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf779
    del primals_1
    del primals_633
    buf781 = buf780[1]
    return (buf781, buf778, buf776, buf775, buf771, buf769, buf768, buf765, buf762, buf759, buf755, buf731, buf752, buf748, buf746, buf745, buf741, buf739, buf738, buf734, buf731, buf730, buf726, buf724, buf723, buf719, buf717, buf716, buf712, buf710, buf708, buf704, buf701, buf696, buf692, buf670, buf690, buf686, buf684, buf683, buf679, buf677, buf676, buf672, buf670, buf668, buf664, buf662, buf661, buf657, buf655, buf654, buf651, buf648, buf647, buf643, buf641, buf639, buf635, buf633, buf632, buf628, buf626, buf625, buf621, buf618, buf617, buf613, buf611, buf610, buf606, buf604, buf603, buf599, buf597, buf595, buf591, buf588, buf587, buf583, buf581, buf580, buf576, buf574, buf573, buf569, buf567, buf565, buf561, buf559, buf558, buf554, buf552, buf551, buf548, buf545, buf544, buf540, buf538, buf536, buf532, buf530, buf529, buf525, buf523, buf522, buf518, buf515, buf514, buf510, buf508, buf507, buf503, buf501, buf500, buf496, buf494, buf492, buf488, buf485, buf480, buf476, buf454, buf474, buf470, buf468, buf467, buf463, buf461, buf460, buf456, buf454, buf452, buf448, buf446, buf445, buf441, buf439, buf438, buf435, buf432, buf431, buf427, buf425, buf423, buf419, buf417, buf416, buf412, buf410, buf409, buf405, buf402, buf401, buf397, buf395, buf394, buf390, buf388, buf387, buf383, buf381, buf379, buf375, buf372, buf371, buf367, buf365, buf364, buf360, buf358, buf357, buf353, buf351, buf349, buf345, buf343, buf342, buf338, buf336, buf335, buf332, buf329, buf328, buf324, buf322, buf320, buf316, buf314, buf313, buf309, buf307, buf306, buf302, buf299, buf298, buf294, buf292, buf291, buf287, buf285, buf284, buf280, buf278, buf276, buf272, buf269, buf268, buf264, buf262, buf261, buf257, buf255, buf254, buf250, buf248, buf246, buf242, buf240, buf239, buf235, buf233, buf232, buf229, buf226, buf225, buf221, buf219, buf217, buf213, buf211, buf210, buf206, buf204, buf203, buf199, buf196, buf195, buf191, buf189, buf188, buf184, buf182, buf181, buf177, buf175, buf173, buf169, buf166, buf165, buf161, buf159, buf158, buf154, buf152, buf151, buf147, buf145, buf143, buf139, buf137, buf136, buf132, buf130, buf129, buf126, buf123, buf122, buf118, buf116, buf114, buf110, buf108, buf107, buf103, buf101, buf100, buf96, buf93, buf92, buf88, buf86, buf85, buf81, buf79, buf78, buf74, buf72, buf70, buf67, buf64, buf60, buf56, buf34, buf54, buf50, buf48, buf47, buf43, buf41, buf40, buf36, buf34, buf32, buf28, buf26, buf25, buf21, buf19, buf18, buf14, buf11, buf10, buf6, buf4, buf2, buf3, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((16, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((256, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((512, 2816, 1, 1), (2816, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((1024, 2560, 1, 1), (2560, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((1000, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_633 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.int64)
    convolution_3 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    getitem_24 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    getitem_25 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.int64)
    convolution_11 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    getitem_88 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    getitem_89 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.int64)
    convolution_40 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_38 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_39 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_40 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_41 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_42 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cat_5 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_44 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_46 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_47 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_48 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_49 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cat_6 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_51 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_52 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_53 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_54 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_55 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_56 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cat_7 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_58 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_187 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_59 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_190 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_60 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_193 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_61 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_196 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_62 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_199 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_63 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_202 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cat_8 = rand_strided((8, 2048, 14, 14), (401408, 1, 28672, 2048), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_205 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_65 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_208 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_66 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_211 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_67 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_214 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_68 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_217 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_69 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_220 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_70 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_223 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cat_9 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_226 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_72 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_229 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_73 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_232 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_74 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_78 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_235 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_75 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_238 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_76 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_241 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_77 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_81 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_244 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cat_10 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    convolution_82 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_247 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_79 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_83 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_250 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_80 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_84 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_253 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_81 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_85 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_256 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_82 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_86 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_259 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_83 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_87 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_262 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_84 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_88 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_265 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cat_11 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_89 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_268 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_86 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_90 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_271 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_87 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_91 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_274 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_88 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_92 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_277 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_89 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_93 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_280 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_90 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_94 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    squeeze_283 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_91 = rand_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    convolution_95 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_286 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cat_12 = rand_strided((8, 2816, 14, 14), (551936, 1, 39424, 2816), device='cpu', dtype=torch.float32)
    convolution_96 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_289 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_93 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_210 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    getitem_211 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.int64)
    convolution_97 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    squeeze_292 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_98 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_295 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_94 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_99 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    squeeze_298 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_95 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    convolution_100 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    squeeze_301 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_96 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_101 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    squeeze_304 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_97 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    convolution_102 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    squeeze_307 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_98 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    convolution_103 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    squeeze_310 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    cat_13 = rand_strided((8, 2560, 7, 7), (125440, 1, 17920, 2560), device='cpu', dtype=torch.float32)
    convolution_104 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    squeeze_313 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.bool)
    unsqueeze_422 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_1 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.bool)
    unsqueeze_434 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_446 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_470 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_494 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_518 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_8 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_530 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_542 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_554 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_566 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_578 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_590 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_602 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_15 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_614 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_626 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_638 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_650 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_662 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_674 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_686 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_22 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_698 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_710 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_722 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_734 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_770 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_29 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_782 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_794 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_806 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_818 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_830 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_842 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_854 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_36 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_866 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_878 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_890 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_902 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_914 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_926 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_938 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_43 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_950 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_962 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_974 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_986 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_998 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1010 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1022 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_50 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_1034 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1046 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1058 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1070 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1082 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1094 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1106 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_57 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_1118 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1130 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1142 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1154 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1166 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1178 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1190 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1202 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_64 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.bool)
    unsqueeze_1214 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1226 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1238 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1250 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1262 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1274 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1286 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_71 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.bool)
    unsqueeze_1298 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1310 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1322 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1334 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1346 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1358 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1370 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_78 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.bool)
    unsqueeze_1382 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1394 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1406 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1418 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1430 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1442 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1454 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_85 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.bool)
    unsqueeze_1466 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1478 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1490 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1502 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1514 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1526 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1538 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1550 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_92 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.bool)
    unsqueeze_1562 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1574 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1586 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1598 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1610 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1622 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1634 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1646 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1658 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1670 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, primals_314, primals_316, primals_633, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, cat, convolution_10, squeeze_31, relu_9, getitem_24, getitem_25, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_10, convolution_13, squeeze_40, relu_11, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, cat_1, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, cat_2, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, relu_24, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, relu_26, convolution_29, squeeze_88, relu_27, convolution_30, squeeze_91, relu_28, convolution_31, squeeze_94, cat_3, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, relu_31, convolution_34, squeeze_103, relu_32, convolution_35, squeeze_106, relu_33, convolution_36, squeeze_109, relu_34, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, cat_4, convolution_39, squeeze_118, relu_37, getitem_88, getitem_89, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, relu_42, convolution_46, squeeze_139, cat_5, convolution_47, squeeze_142, relu_44, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, relu_46, convolution_50, squeeze_151, relu_47, convolution_51, squeeze_154, relu_48, convolution_52, squeeze_157, relu_49, convolution_53, squeeze_160, cat_6, convolution_54, squeeze_163, relu_51, convolution_55, squeeze_166, relu_52, convolution_56, squeeze_169, relu_53, convolution_57, squeeze_172, relu_54, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, relu_56, convolution_60, squeeze_181, cat_7, convolution_61, squeeze_184, relu_58, convolution_62, squeeze_187, relu_59, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, relu_61, convolution_65, squeeze_196, relu_62, convolution_66, squeeze_199, relu_63, convolution_67, squeeze_202, cat_8, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, relu_66, convolution_70, squeeze_211, relu_67, convolution_71, squeeze_214, relu_68, convolution_72, squeeze_217, relu_69, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, cat_9, convolution_75, squeeze_226, relu_72, convolution_76, squeeze_229, relu_73, convolution_77, squeeze_232, relu_74, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, relu_76, convolution_80, squeeze_241, relu_77, convolution_81, squeeze_244, cat_10, convolution_82, squeeze_247, relu_79, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, relu_81, convolution_85, squeeze_256, relu_82, convolution_86, squeeze_259, relu_83, convolution_87, squeeze_262, relu_84, convolution_88, squeeze_265, cat_11, convolution_89, squeeze_268, relu_86, convolution_90, squeeze_271, relu_87, convolution_91, squeeze_274, relu_88, convolution_92, squeeze_277, relu_89, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, relu_91, convolution_95, squeeze_286, cat_12, convolution_96, squeeze_289, relu_93, getitem_210, getitem_211, convolution_97, squeeze_292, convolution_98, squeeze_295, relu_94, convolution_99, squeeze_298, relu_95, convolution_100, squeeze_301, relu_96, convolution_101, squeeze_304, relu_97, convolution_102, squeeze_307, relu_98, convolution_103, squeeze_310, cat_13, convolution_104, squeeze_313, clone, le, unsqueeze_422, le_1, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, le_8, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, le_15, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, le_22, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, le_29, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, le_36, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, unsqueeze_914, unsqueeze_926, unsqueeze_938, le_43, unsqueeze_950, unsqueeze_962, unsqueeze_974, unsqueeze_986, unsqueeze_998, unsqueeze_1010, unsqueeze_1022, le_50, unsqueeze_1034, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1082, unsqueeze_1094, unsqueeze_1106, le_57, unsqueeze_1118, unsqueeze_1130, unsqueeze_1142, unsqueeze_1154, unsqueeze_1166, unsqueeze_1178, unsqueeze_1190, unsqueeze_1202, le_64, unsqueeze_1214, unsqueeze_1226, unsqueeze_1238, unsqueeze_1250, unsqueeze_1262, unsqueeze_1274, unsqueeze_1286, le_71, unsqueeze_1298, unsqueeze_1310, unsqueeze_1322, unsqueeze_1334, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, le_78, unsqueeze_1382, unsqueeze_1394, unsqueeze_1406, unsqueeze_1418, unsqueeze_1430, unsqueeze_1442, unsqueeze_1454, le_85, unsqueeze_1466, unsqueeze_1478, unsqueeze_1490, unsqueeze_1502, unsqueeze_1514, unsqueeze_1526, unsqueeze_1538, unsqueeze_1550, le_92, unsqueeze_1562, unsqueeze_1574, unsqueeze_1586, unsqueeze_1598, unsqueeze_1610, unsqueeze_1622, unsqueeze_1634, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dla102', benchmark_compiled_module)
