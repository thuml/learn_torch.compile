
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


cpp_fused_convolution_backward_div_leaky_relu_backward_native_batch_norm_backward_sum_0 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (1024L*x2) + (65536L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x2) + (65536L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(64.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(0.01);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp0);
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp12 = tmp8 * tmp11;
                            tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1024L*x1) + (65536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(64.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp0);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp13 = static_cast<float>(0.001953125);
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
                        tmp26.store(out_ptr4 + static_cast<long>(x2 + (1024L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (1024L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1024L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_3 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.001953125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_4 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.01);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.01);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.001953125);
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_5 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.001953125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.01);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.01);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
                    auto tmp12 = tmp10 - tmp11;
                    auto tmp14 = static_cast<float>(0.001953125);
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_7 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.001953125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_8 = async_compile.cpp('''
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
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = static_cast<float>(0.01);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 * tmp9;
                auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp0);
                tmp11.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001953125);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_9 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.001953125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(512);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (65536L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(1024);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr0[static_cast<long>((-512L) + x2 + (512L*x1) + (32768L*x0))];
                            auto tmp13 = in_ptr1[static_cast<long>((-512L) + x2 + (512L*x1) + (32768L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = in_ptr2[static_cast<long>((-512L) + x2 + (512L*x1) + (32768L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>((-512L) + x2 + (512L*x1) + (32768L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr4[static_cast<long>((-32768L) + x1 + (64L*x2) + (32768L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (65536L*x0))] = tmp22;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
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
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.001953125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_12 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_15 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_16 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.01);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.01);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.00048828125);
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_17 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.01);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.01);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
                    auto tmp12 = tmp10 - tmp11;
                    auto tmp14 = static_cast<float>(0.00048828125);
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_19 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_20 = async_compile.cpp('''
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
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = static_cast<float>(0.01);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 * tmp9;
                auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp0);
                tmp11.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00048828125);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_21 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_23 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_24 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.01);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.01);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.00048828125);
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_25 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.01);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.01);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
                    auto tmp12 = tmp10 - tmp11;
                    auto tmp14 = static_cast<float>(0.00048828125);
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_27 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_28 = async_compile.cpp('''
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
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = static_cast<float>(0.01);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 * tmp9;
                auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp0);
                tmp11.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00048828125);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_29 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(256);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (131072L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(512);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr0[static_cast<long>((-256L) + x2 + (256L*x1) + (65536L*x0))];
                            auto tmp13 = in_ptr1[static_cast<long>((-256L) + x2 + (256L*x1) + (65536L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = in_ptr2[static_cast<long>((-256L) + x2 + (256L*x1) + (65536L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>((-256L) + x2 + (256L*x1) + (65536L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr4[static_cast<long>((-65536L) + x1 + (256L*x2) + (65536L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (131072L*x0))] = tmp22;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_31 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00048828125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_32 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_33 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_34 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_35 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.01);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.01);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.0001220703125);
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_37 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.01);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.01);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
                    auto tmp12 = tmp10 - tmp11;
                    auto tmp14 = static_cast<float>(0.0001220703125);
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_39 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_40 = async_compile.cpp('''
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
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = static_cast<float>(0.01);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 * tmp9;
                auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp0);
                tmp11.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0001220703125);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_41 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_43 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.01);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.01);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.0001220703125);
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_45 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.01);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.01);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp0);
                    auto tmp12 = tmp10 - tmp11;
                    auto tmp14 = static_cast<float>(0.0001220703125);
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_47 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_48 = async_compile.cpp('''
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
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = static_cast<float>(0.01);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 * tmp9;
                auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp0);
                tmp11.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0001220703125);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_49 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(128);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_out_ptr0[static_cast<long>(x2 + (256L*x1) + (262144L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(256);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr0[static_cast<long>((-128L) + x2 + (128L*x1) + (131072L*x0))];
                            auto tmp13 = in_ptr1[static_cast<long>((-128L) + x2 + (128L*x1) + (131072L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = in_ptr2[static_cast<long>((-128L) + x2 + (128L*x1) + (131072L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>((-128L) + x2 + (128L*x1) + (131072L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr4[static_cast<long>((-131072L) + x1 + (1024L*x2) + (131072L*x0))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        in_out_ptr0[static_cast<long>(x2 + (256L*x1) + (262144L*x0))] = tmp22;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
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


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_51 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.0001220703125);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_52 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(3.0517578125e-05);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(3.0517578125e-05);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(3.0517578125e-05);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_55 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(3.0517578125e-05);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.01);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(0.01);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = decltype(tmp3)::blendv(tmp6, tmp3, tmp0);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(3.0517578125e-05);
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_57 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(3.0517578125e-05);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_58 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (524288L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(64);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (524288L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(128);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-64L) + x2 + (64L*x1) + (262144L*x0))];
                            auto tmp14 = in_ptr2[static_cast<long>((-64L) + x2 + (64L*x1) + (262144L*x0))];
                            auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                            auto tmp16 = in_ptr3[static_cast<long>((-262144L) + x1 + (4096L*x2) + (262144L*x0))];
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            return tmp17;
                        }
                        ;
                        auto tmp18 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp19 = tmp5 ? tmp8 : tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (524288L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp5 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = [&]
                        {
                            auto tmp24 = in_ptr1[static_cast<long>((-64L) + x2 + (64L*x1) + (262144L*x0))];
                            auto tmp25 = in_ptr2[static_cast<long>((-64L) + x2 + (64L*x1) + (262144L*x0))];
                            auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                            auto tmp27 = in_ptr3[static_cast<long>((-262144L) + x1 + (4096L*x2) + (262144L*x0))];
                            auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                            return tmp28;
                        }
                        ;
                        auto tmp29 = tmp9 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                        auto tmp30 = tmp5 ? tmp22 : tmp29;
                        auto tmp31 = static_cast<float>(0.01);
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        auto tmp33 = tmp0 ? tmp19 : tmp32;
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (524288L*x0))] = tmp33;
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(3.0517578125e-05);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_59 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(3.0517578125e-05);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_60 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(7.62939453125e-06);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(7.62939453125e-06);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.01);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.01);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = decltype(tmp1)::blendv(tmp4, tmp1, tmp0);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(7.62939453125e-06);
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
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(7.62939453125e-06);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (128L*x2) + (2097152L*x1))];
                            auto tmp50 = in_ptr4[static_cast<long>(x0 + (128L*x2) + (2097152L*x1))];
                            auto tmp51 = in_ptr5[static_cast<long>(x0)];
                            auto tmp1 = c10::convert<long>(x0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 >= tmp2;
                            auto tmp4 = static_cast<long>(64);
                            auto tmp5 = tmp1 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>(x0 + (128L*x2) + (2097152L*x1))];
                                return tmp7;
                            }
                            ;
                            auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp9 = tmp1 >= tmp4;
                            auto tmp10 = static_cast<long>(128);
                            auto tmp11 = tmp1 < tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = in_ptr2[static_cast<long>((-64L) + x0 + (64L*x2) + (1048576L*x1))];
                                auto tmp14 = in_ptr3[static_cast<long>((-1048576L) + x2 + (16384L*x0) + (1048576L*x1))];
                                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                                return tmp15;
                            }
                            ;
                            auto tmp16 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                            auto tmp17 = tmp5 ? tmp8 : tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = in_ptr1[static_cast<long>(x0 + (128L*x2) + (2097152L*x1))];
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp5 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr2[static_cast<long>((-64L) + x0 + (64L*x2) + (1048576L*x1))];
                                auto tmp23 = in_ptr3[static_cast<long>((-1048576L) + x2 + (16384L*x0) + (1048576L*x1))];
                                auto tmp24 = decltype(tmp22)(tmp22 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp9 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp26 = tmp5 ? tmp20 : tmp25;
                            auto tmp27 = static_cast<float>(0.01);
                            auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                            auto tmp29 = tmp0 ? tmp17 : tmp28;
                            auto tmp30 = [&]
                            {
                                auto tmp31 = in_ptr1[static_cast<long>(x0 + (128L*x2) + (2097152L*x1))];
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp5 ? tmp30() : static_cast<decltype(tmp30())>(0.0);
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr2[static_cast<long>((-64L) + x0 + (64L*x2) + (1048576L*x1))];
                                auto tmp35 = in_ptr3[static_cast<long>((-1048576L) + x2 + (16384L*x0) + (1048576L*x1))];
                                auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                                return tmp36;
                            }
                            ;
                            auto tmp37 = tmp9 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp38 = tmp5 ? tmp32 : tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = in_ptr1[static_cast<long>(x0 + (128L*x2) + (2097152L*x1))];
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp5 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp42 = [&]
                            {
                                auto tmp43 = in_ptr2[static_cast<long>((-64L) + x0 + (64L*x2) + (1048576L*x1))];
                                auto tmp44 = in_ptr3[static_cast<long>((-1048576L) + x2 + (16384L*x0) + (1048576L*x1))];
                                auto tmp45 = decltype(tmp43)(tmp43 + tmp44);
                                return tmp45;
                            }
                            ;
                            auto tmp46 = tmp9 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                            auto tmp47 = tmp5 ? tmp41 : tmp46;
                            auto tmp48 = decltype(tmp47)(tmp47 * tmp27);
                            auto tmp49 = tmp0 ? tmp38 : tmp48;
                            auto tmp52 = decltype(tmp50)(tmp50 - tmp51);
                            auto tmp53 = decltype(tmp49)(tmp49 * tmp52);
                            tmp_acc0 = tmp_acc0 + tmp29;
                            tmp_acc1 = tmp_acc1 + tmp53;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (2097152L*x0))];
                        auto tmp30 = in_ptr4[static_cast<long>(x2 + (128L*x1) + (2097152L*x0))];
                        auto tmp31 = in_ptr5[static_cast<long>(x2)];
                        auto tmp33 = out_ptr1[static_cast<long>(x2)];
                        auto tmp36 = in_ptr6[static_cast<long>(x2)];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(64);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (2097152L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(128);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-64L) + x2 + (64L*x1) + (1048576L*x0))];
                            auto tmp14 = in_ptr3[static_cast<long>((-1048576L) + x1 + (16384L*x2) + (1048576L*x0))];
                            auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp17 = tmp5 ? tmp8 : tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (2097152L*x0))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp5 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = [&]
                        {
                            auto tmp22 = in_ptr2[static_cast<long>((-64L) + x2 + (64L*x1) + (1048576L*x0))];
                            auto tmp23 = in_ptr3[static_cast<long>((-1048576L) + x1 + (16384L*x2) + (1048576L*x0))];
                            auto tmp24 = decltype(tmp22)(tmp22 + tmp23);
                            return tmp24;
                        }
                        ;
                        auto tmp25 = tmp9 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp26 = tmp5 ? tmp20 : tmp25;
                        auto tmp27 = static_cast<float>(0.01);
                        auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                        auto tmp29 = tmp0 ? tmp17 : tmp28;
                        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                        auto tmp34 = static_cast<float>(7.62939453125e-06);
                        auto tmp35 = decltype(tmp33)(tmp33 * tmp34);
                        auto tmp37 = decltype(tmp36)(tmp36 * tmp36);
                        auto tmp38 = decltype(tmp35)(tmp35 * tmp37);
                        auto tmp39 = decltype(tmp32)(tmp32 * tmp38);
                        auto tmp40 = decltype(tmp29)(tmp29 - tmp39);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (2097152L*x0))] = tmp40;
                    }
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(7.62939453125e-06);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 - tmp4;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_65 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(7.62939453125e-06);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(524288L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.01);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.01);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(1.9073486328125e-06);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_405, convolution, squeeze_1, where, convolution_1, squeeze_4, where_1, convolution_2, squeeze_7, getitem_9, convolution_3, squeeze_10, where_3, convolution_4, squeeze_13, add_25, convolution_5, squeeze_16, cat, convolution_6, squeeze_19, where_6, convolution_7, squeeze_22, where_7, convolution_8, squeeze_25, getitem_27, convolution_9, squeeze_28, where_9, convolution_10, squeeze_31, add_56, convolution_11, squeeze_34, where_11, convolution_12, squeeze_37, add_67, convolution_13, squeeze_40, cat_1, convolution_14, squeeze_43, where_14, convolution_15, squeeze_46, where_15, convolution_16, squeeze_49, getitem_49, convolution_17, squeeze_52, where_17, convolution_18, squeeze_55, add_98, convolution_19, squeeze_58, where_19, convolution_20, squeeze_61, add_109, convolution_21, squeeze_64, where_21, convolution_22, squeeze_67, add_120, convolution_23, squeeze_70, where_23, convolution_24, squeeze_73, add_131, convolution_25, squeeze_76, where_25, convolution_26, squeeze_79, add_142, convolution_27, squeeze_82, where_27, convolution_28, squeeze_85, add_153, convolution_29, squeeze_88, where_29, convolution_30, squeeze_91, add_164, convolution_31, squeeze_94, where_31, convolution_32, squeeze_97, add_175, convolution_33, squeeze_100, cat_2, convolution_34, squeeze_103, where_34, convolution_35, squeeze_106, where_35, convolution_36, squeeze_109, getitem_95, convolution_37, squeeze_112, where_37, convolution_38, squeeze_115, add_206, convolution_39, squeeze_118, where_39, convolution_40, squeeze_121, add_217, convolution_41, squeeze_124, where_41, convolution_42, squeeze_127, add_228, convolution_43, squeeze_130, where_43, convolution_44, squeeze_133, add_239, convolution_45, squeeze_136, where_45, convolution_46, squeeze_139, add_250, convolution_47, squeeze_142, where_47, convolution_48, squeeze_145, add_261, convolution_49, squeeze_148, where_49, convolution_50, squeeze_151, add_272, convolution_51, squeeze_154, where_51, convolution_52, squeeze_157, add_283, convolution_53, squeeze_160, cat_3, convolution_54, squeeze_163, where_54, convolution_55, squeeze_166, where_55, convolution_56, squeeze_169, getitem_141, convolution_57, squeeze_172, where_57, convolution_58, squeeze_175, add_314, convolution_59, squeeze_178, where_59, convolution_60, squeeze_181, add_325, convolution_61, squeeze_184, where_61, convolution_62, squeeze_187, add_336, convolution_63, squeeze_190, where_63, convolution_64, squeeze_193, add_347, convolution_65, squeeze_196, cat_4, convolution_66, squeeze_199, clone, permute_1, gt_67, unsqueeze_270, gt_68, unsqueeze_282, gt_69, unsqueeze_294, unsqueeze_306, gt_71, unsqueeze_318, unsqueeze_330, gt_73, unsqueeze_342, unsqueeze_354, gt_75, unsqueeze_366, unsqueeze_378, gt_77, unsqueeze_390, unsqueeze_402, unsqueeze_414, gt_80, unsqueeze_426, gt_81, unsqueeze_438, unsqueeze_450, gt_83, unsqueeze_462, unsqueeze_474, gt_85, unsqueeze_486, unsqueeze_498, gt_87, unsqueeze_510, unsqueeze_522, gt_89, unsqueeze_534, unsqueeze_546, gt_91, unsqueeze_558, unsqueeze_570, gt_93, unsqueeze_582, unsqueeze_594, gt_95, unsqueeze_606, unsqueeze_618, gt_97, unsqueeze_630, unsqueeze_642, unsqueeze_654, gt_100, unsqueeze_666, gt_101, unsqueeze_678, unsqueeze_690, gt_103, unsqueeze_702, unsqueeze_714, gt_105, unsqueeze_726, unsqueeze_738, gt_107, unsqueeze_750, unsqueeze_762, gt_109, unsqueeze_774, unsqueeze_786, gt_111, unsqueeze_798, unsqueeze_810, gt_113, unsqueeze_822, unsqueeze_834, gt_115, unsqueeze_846, unsqueeze_858, gt_117, unsqueeze_870, unsqueeze_882, unsqueeze_894, gt_120, unsqueeze_906, gt_121, unsqueeze_918, unsqueeze_930, gt_123, unsqueeze_942, unsqueeze_954, gt_125, unsqueeze_966, unsqueeze_978, unsqueeze_990, gt_128, unsqueeze_1002, gt_129, unsqueeze_1014, unsqueeze_1026, gt_131, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_97, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_113, (1024, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_129, (512, ), (1, ))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_135, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_136, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_137, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_138, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_139, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_140, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_141, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_142, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_143, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_144, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_145, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_146, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_147, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_148, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_149, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_150, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_151, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_152, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_153, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_154, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_155, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_156, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_157, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_158, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_159, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_160, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_161, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_162, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_163, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_164, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_165, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_166, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_167, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_168, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_169, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_170, (512, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_171, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_172, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_173, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_174, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_175, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_176, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_177, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_178, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_179, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_180, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_181, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_182, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_183, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_184, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_185, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_186, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_187, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_188, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_189, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_190, (1024, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_191, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_192, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_193, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_194, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_195, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_196, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_197, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_198, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_199, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_200, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_201, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_405, (8, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(convolution, (8, 32, 256, 256), (2097152, 1, 8192, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(where, (8, 32, 256, 256), (2097152, 1, 8192, 32))
    assert_size_stride(convolution_1, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(where_1, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(convolution_2, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(squeeze_7, (128, ), (1, ))
    assert_size_stride(getitem_9, (8, 64, 128, 128), (2097152, 16384, 128, 1))
    assert_size_stride(convolution_3, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(squeeze_10, (32, ), (1, ))
    assert_size_stride(where_3, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(convolution_4, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(add_25, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(convolution_5, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(cat, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(convolution_6, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(where_6, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(convolution_7, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_22, (128, ), (1, ))
    assert_size_stride(where_7, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_8, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_25, (128, ), (1, ))
    assert_size_stride(getitem_27, (8, 64, 64, 64), (524288, 4096, 64, 1))
    assert_size_stride(convolution_9, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_28, (64, ), (1, ))
    assert_size_stride(where_9, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_10, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_31, (64, ), (1, ))
    assert_size_stride(add_56, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_11, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_34, (64, ), (1, ))
    assert_size_stride(where_11, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_12, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_37, (64, ), (1, ))
    assert_size_stride(add_67, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_13, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_40, (64, ), (1, ))
    assert_size_stride(cat_1, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_14, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_43, (128, ), (1, ))
    assert_size_stride(where_14, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_15, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_46, (256, ), (1, ))
    assert_size_stride(where_15, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_16, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_49, (256, ), (1, ))
    assert_size_stride(getitem_49, (8, 128, 32, 32), (262144, 1024, 32, 1))
    assert_size_stride(convolution_17, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_52, (128, ), (1, ))
    assert_size_stride(where_17, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_18, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_55, (128, ), (1, ))
    assert_size_stride(add_98, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_19, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(where_19, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_20, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_61, (128, ), (1, ))
    assert_size_stride(add_109, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_21, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(where_21, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_22, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(add_120, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_23, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_70, (128, ), (1, ))
    assert_size_stride(where_23, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_24, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_73, (128, ), (1, ))
    assert_size_stride(add_131, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_25, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_76, (128, ), (1, ))
    assert_size_stride(where_25, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_26, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_79, (128, ), (1, ))
    assert_size_stride(add_142, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_27, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_82, (128, ), (1, ))
    assert_size_stride(where_27, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_28, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_85, (128, ), (1, ))
    assert_size_stride(add_153, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_29, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_88, (128, ), (1, ))
    assert_size_stride(where_29, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_30, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_91, (128, ), (1, ))
    assert_size_stride(add_164, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_31, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_94, (128, ), (1, ))
    assert_size_stride(where_31, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_32, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_97, (128, ), (1, ))
    assert_size_stride(add_175, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_33, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_100, (128, ), (1, ))
    assert_size_stride(cat_2, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_34, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_103, (256, ), (1, ))
    assert_size_stride(where_34, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_35, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_106, (512, ), (1, ))
    assert_size_stride(where_35, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(convolution_36, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_109, (512, ), (1, ))
    assert_size_stride(getitem_95, (8, 256, 16, 16), (131072, 256, 16, 1))
    assert_size_stride(convolution_37, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_112, (256, ), (1, ))
    assert_size_stride(where_37, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_38, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(add_206, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_39, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_118, (256, ), (1, ))
    assert_size_stride(where_39, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_40, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_121, (256, ), (1, ))
    assert_size_stride(add_217, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_41, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_124, (256, ), (1, ))
    assert_size_stride(where_41, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_42, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_127, (256, ), (1, ))
    assert_size_stride(add_228, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_43, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_130, (256, ), (1, ))
    assert_size_stride(where_43, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_44, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_133, (256, ), (1, ))
    assert_size_stride(add_239, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_45, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_136, (256, ), (1, ))
    assert_size_stride(where_45, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_46, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_139, (256, ), (1, ))
    assert_size_stride(add_250, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_47, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_142, (256, ), (1, ))
    assert_size_stride(where_47, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_48, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_145, (256, ), (1, ))
    assert_size_stride(add_261, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_49, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_148, (256, ), (1, ))
    assert_size_stride(where_49, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_50, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_151, (256, ), (1, ))
    assert_size_stride(add_272, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_51, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_154, (256, ), (1, ))
    assert_size_stride(where_51, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_52, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_157, (256, ), (1, ))
    assert_size_stride(add_283, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_53, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_160, (256, ), (1, ))
    assert_size_stride(cat_3, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(convolution_54, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_163, (512, ), (1, ))
    assert_size_stride(where_54, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(convolution_55, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(squeeze_166, (1024, ), (1, ))
    assert_size_stride(where_55, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(convolution_56, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(squeeze_169, (1024, ), (1, ))
    assert_size_stride(getitem_141, (8, 512, 8, 8), (65536, 64, 8, 1))
    assert_size_stride(convolution_57, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_172, (512, ), (1, ))
    assert_size_stride(where_57, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_58, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_175, (512, ), (1, ))
    assert_size_stride(add_314, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_59, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_178, (512, ), (1, ))
    assert_size_stride(where_59, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_60, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_181, (512, ), (1, ))
    assert_size_stride(add_325, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_61, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_184, (512, ), (1, ))
    assert_size_stride(where_61, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_62, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_187, (512, ), (1, ))
    assert_size_stride(add_336, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_63, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_190, (512, ), (1, ))
    assert_size_stride(where_63, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_64, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_193, (512, ), (1, ))
    assert_size_stride(add_347, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_65, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_196, (512, ), (1, ))
    assert_size_stride(cat_4, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(convolution_66, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(squeeze_199, (1024, ), (1, ))
    assert_size_stride(clone, (8, 1024), (1024, 1))
    assert_size_stride(permute_1, (1000, 1024), (1024, 1))
    assert_size_stride(gt_67, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(unsqueeze_270, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(gt_68, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_282, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_69, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_294, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_71, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_318, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_73, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_342, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_75, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_366, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_77, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    assert_size_stride(unsqueeze_390, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(gt_80, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_426, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_81, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_438, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_83, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_462, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_85, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_486, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_87, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_510, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_522, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_89, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_534, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_91, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_558, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_93, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_582, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_594, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_95, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_606, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_97, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(unsqueeze_630, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(gt_100, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_666, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_101, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_678, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_103, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_702, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_105, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_726, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_738, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_107, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_750, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_109, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_774, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_786, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_111, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_798, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_810, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_113, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_822, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_115, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_846, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_858, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_117, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(unsqueeze_870, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_882, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_894, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(gt_120, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_906, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_121, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_918, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_930, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_123, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_942, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_954, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_125, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(unsqueeze_966, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_978, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_990, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_128, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(unsqueeze_1002, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(gt_129, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(unsqueeze_1014, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1026, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(gt_131, (8, 128, 128, 128), (2097152, 1, 16384, 128))
    assert_size_stride(unsqueeze_1038, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1050, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1062, (1, 32, 1, 1), (32, 1, 1, 1))
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
    buf3 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf4 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf5 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_leaky_relu_backward_native_batch_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(gt_67.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(unsqueeze_270.data_ptr()), c_void_p(squeeze_199.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del convolution_66
    del gt_67
    del primals_133
    del squeeze_199
    del tangents_1
    del unsqueeze_270
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf7 = aten.convolution_backward(buf6, cat_4, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf6
    del cat_4
    del primals_201
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((512, ), device='cpu', dtype=torch.float32)
    buf11 = empty((512, ), device='cpu', dtype=torch.float32)
    buf12 = empty((512, ), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_1(c_void_p(gt_68.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(squeeze_196.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del convolution_65
    del gt_68
    del primals_131
    del squeeze_196
    del unsqueeze_282
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf14 = aten.convolution_backward(buf13, add_347, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_347
    del primals_200
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = buf11; del buf11  # reuse
    buf18 = empty((512, ), device='cpu', dtype=torch.float32)
    buf19 = empty((512, ), device='cpu', dtype=torch.float32)
    buf20 = buf13; del buf13  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_2(c_void_p(gt_69.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_294.data_ptr()), c_void_p(squeeze_193.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    del convolution_64
    del gt_69
    del primals_129
    del squeeze_193
    del unsqueeze_294
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf21 = aten.convolution_backward(buf20, where_63, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf20
    del primals_199
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    buf24 = buf18; del buf18  # reuse
    buf25 = empty((512, ), device='cpu', dtype=torch.float32)
    buf26 = empty((512, ), device='cpu', dtype=torch.float32)
    buf27 = buf22; del buf22  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_3(c_void_p(buf27.data_ptr()), c_void_p(where_63.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_190.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del convolution_63
    del primals_127
    del squeeze_190
    del unsqueeze_306
    del where_63
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf28 = aten.convolution_backward(buf27, add_336, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_336
    del primals_198
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf31 = buf25; del buf25  # reuse
    buf32 = empty((512, ), device='cpu', dtype=torch.float32)
    buf33 = buf27; del buf27  # reuse
    buf34 = buf32; del buf32  # reuse
    cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_4(c_void_p(buf34.data_ptr()), c_void_p(gt_71.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(unsqueeze_318.data_ptr()), c_void_p(squeeze_187.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()))
    del convolution_62
    del gt_71
    del primals_125
    del squeeze_187
    del unsqueeze_318
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf35 = aten.convolution_backward(buf33, where_61, primals_197, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf33
    del primals_197
    buf36 = buf35[0]
    buf37 = buf35[1]
    del buf35
    buf38 = empty((512, ), device='cpu', dtype=torch.float32)
    buf39 = empty((512, ), device='cpu', dtype=torch.float32)
    buf40 = empty((512, ), device='cpu', dtype=torch.float32)
    buf41 = buf36; del buf36  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_5(c_void_p(buf41.data_ptr()), c_void_p(where_61.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    del convolution_61
    del primals_123
    del squeeze_184
    del unsqueeze_330
    del where_61
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf42 = aten.convolution_backward(buf41, add_325, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_325
    del primals_196
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    buf45 = buf39; del buf39  # reuse
    buf46 = empty((512, ), device='cpu', dtype=torch.float32)
    buf47 = buf41; del buf41  # reuse
    buf49 = buf47; del buf47  # reuse
    buf48 = buf46; del buf46  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_6(c_void_p(buf49.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(gt_73.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf45.data_ptr()))
    del convolution_60
    del gt_73
    del primals_121
    del squeeze_181
    del unsqueeze_342
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf50 = aten.convolution_backward(buf49, where_59, primals_195, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf49
    del primals_195
    buf51 = buf50[0]
    buf52 = buf50[1]
    del buf50
    buf53 = empty((512, ), device='cpu', dtype=torch.float32)
    buf54 = empty((512, ), device='cpu', dtype=torch.float32)
    buf55 = empty((512, ), device='cpu', dtype=torch.float32)
    buf56 = buf51; del buf51  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_7(c_void_p(buf56.data_ptr()), c_void_p(where_59.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del convolution_59
    del primals_119
    del squeeze_178
    del unsqueeze_354
    del where_59
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf57 = aten.convolution_backward(buf56, add_314, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_314
    del primals_194
    buf58 = buf57[0]
    buf59 = buf57[1]
    del buf57
    buf60 = buf56; del buf56  # reuse
    buf61 = buf54; del buf54  # reuse
    buf62 = empty((512, ), device='cpu', dtype=torch.float32)
    buf63 = empty((512, ), device='cpu', dtype=torch.float32)
    buf64 = buf60; del buf60  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_8(c_void_p(buf64.data_ptr()), c_void_p(gt_75.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    del convolution_58
    del gt_75
    del primals_117
    del squeeze_175
    del unsqueeze_366
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf65 = aten.convolution_backward(buf64, where_57, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf64
    del primals_193
    buf66 = buf65[0]
    buf67 = buf65[1]
    del buf65
    buf68 = buf62; del buf62  # reuse
    buf69 = empty((512, ), device='cpu', dtype=torch.float32)
    buf70 = empty((512, ), device='cpu', dtype=torch.float32)
    buf71 = buf66; del buf66  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_9(c_void_p(buf71.data_ptr()), c_void_p(where_57.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del convolution_57
    del primals_115
    del squeeze_172
    del unsqueeze_378
    del where_57
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf72 = aten.convolution_backward(buf71, getitem_141, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf71
    del getitem_141
    del primals_192
    buf73 = buf72[0]
    buf74 = buf72[1]
    del buf72
    buf75 = buf8; del buf8  # reuse
    buf76 = buf4; del buf4  # reuse
    buf77 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf78 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf79 = buf75; del buf75  # reuse
    cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_10(c_void_p(buf79.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(gt_77.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del buf15
    del buf29
    del buf43
    del buf58
    del buf73
    del convolution_56
    del gt_77
    del primals_113
    del squeeze_169
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf80 = aten.convolution_backward(buf79, where_55, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf79
    del primals_191
    buf81 = buf80[0]
    buf82 = buf80[1]
    del buf80
    buf83 = buf77; del buf77  # reuse
    buf84 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf85 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf86 = buf81; del buf81  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_11(c_void_p(buf86.data_ptr()), c_void_p(where_55.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    del buf84
    del convolution_55
    del primals_111
    del squeeze_166
    del unsqueeze_402
    del where_55
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf87 = aten.convolution_backward(buf86, where_54, primals_190, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_190
    buf88 = buf87[0]
    buf89 = buf87[1]
    del buf87
    buf90 = buf69; del buf69  # reuse
    buf91 = empty((512, ), device='cpu', dtype=torch.float32)
    buf92 = empty((512, ), device='cpu', dtype=torch.float32)
    buf93 = buf88; del buf88  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_12(c_void_p(buf93.data_ptr()), c_void_p(where_54.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    del convolution_54
    del primals_109
    del squeeze_163
    del unsqueeze_414
    del where_54
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf94 = aten.convolution_backward(buf93, cat_3, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf93
    del cat_3
    del primals_189
    buf95 = buf94[0]
    buf96 = buf94[1]
    del buf94
    buf97 = empty((256, ), device='cpu', dtype=torch.float32)
    buf98 = empty((256, ), device='cpu', dtype=torch.float32)
    buf99 = empty((256, ), device='cpu', dtype=torch.float32)
    buf100 = reinterpret_tensor(buf86, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf86  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_13(c_void_p(gt_80.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()))
    del convolution_53
    del gt_80
    del primals_107
    del squeeze_160
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf101 = aten.convolution_backward(buf100, add_283, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_283
    del primals_188
    buf102 = buf101[0]
    buf103 = buf101[1]
    del buf101
    buf104 = buf98; del buf98  # reuse
    buf105 = empty((256, ), device='cpu', dtype=torch.float32)
    buf106 = empty((256, ), device='cpu', dtype=torch.float32)
    buf107 = buf100; del buf100  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_14(c_void_p(gt_81.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    del convolution_52
    del gt_81
    del primals_105
    del squeeze_157
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf108 = aten.convolution_backward(buf107, where_51, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf107
    del primals_187
    buf109 = buf108[0]
    buf110 = buf108[1]
    del buf108
    buf111 = buf105; del buf105  # reuse
    buf112 = empty((256, ), device='cpu', dtype=torch.float32)
    buf113 = empty((256, ), device='cpu', dtype=torch.float32)
    buf114 = buf109; del buf109  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_15(c_void_p(buf114.data_ptr()), c_void_p(where_51.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    del convolution_51
    del primals_103
    del squeeze_154
    del unsqueeze_450
    del where_51
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf115 = aten.convolution_backward(buf114, add_272, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_272
    del primals_186
    buf116 = buf115[0]
    buf117 = buf115[1]
    del buf115
    buf118 = buf112; del buf112  # reuse
    buf119 = empty((256, ), device='cpu', dtype=torch.float32)
    buf120 = buf114; del buf114  # reuse
    buf121 = buf119; del buf119  # reuse
    cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_16(c_void_p(buf121.data_ptr()), c_void_p(gt_83.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()))
    del convolution_50
    del gt_83
    del primals_101
    del squeeze_151
    del unsqueeze_462
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf122 = aten.convolution_backward(buf120, where_49, primals_185, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf120
    del primals_185
    buf123 = buf122[0]
    buf124 = buf122[1]
    del buf122
    buf125 = empty((256, ), device='cpu', dtype=torch.float32)
    buf126 = empty((256, ), device='cpu', dtype=torch.float32)
    buf127 = empty((256, ), device='cpu', dtype=torch.float32)
    buf128 = buf123; del buf123  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_17(c_void_p(buf128.data_ptr()), c_void_p(where_49.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del convolution_49
    del primals_99
    del squeeze_148
    del unsqueeze_474
    del where_49
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf129 = aten.convolution_backward(buf128, add_261, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_261
    del primals_184
    buf130 = buf129[0]
    buf131 = buf129[1]
    del buf129
    buf132 = buf126; del buf126  # reuse
    buf133 = empty((256, ), device='cpu', dtype=torch.float32)
    buf134 = buf128; del buf128  # reuse
    buf136 = buf134; del buf134  # reuse
    buf135 = buf133; del buf133  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_18(c_void_p(buf136.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(gt_85.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf132.data_ptr()))
    del convolution_48
    del gt_85
    del primals_97
    del squeeze_145
    del unsqueeze_486
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf137 = aten.convolution_backward(buf136, where_47, primals_183, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf136
    del primals_183
    buf138 = buf137[0]
    buf139 = buf137[1]
    del buf137
    buf140 = empty((256, ), device='cpu', dtype=torch.float32)
    buf141 = empty((256, ), device='cpu', dtype=torch.float32)
    buf142 = empty((256, ), device='cpu', dtype=torch.float32)
    buf143 = buf138; del buf138  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_19(c_void_p(buf143.data_ptr()), c_void_p(where_47.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    del convolution_47
    del primals_95
    del squeeze_142
    del unsqueeze_498
    del where_47
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf144 = aten.convolution_backward(buf143, add_250, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_250
    del primals_182
    buf145 = buf144[0]
    buf146 = buf144[1]
    del buf144
    buf147 = buf143; del buf143  # reuse
    buf148 = buf141; del buf141  # reuse
    buf149 = empty((256, ), device='cpu', dtype=torch.float32)
    buf150 = empty((256, ), device='cpu', dtype=torch.float32)
    buf151 = buf147; del buf147  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_20(c_void_p(buf151.data_ptr()), c_void_p(gt_87.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    del convolution_46
    del gt_87
    del primals_93
    del squeeze_139
    del unsqueeze_510
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf152 = aten.convolution_backward(buf151, where_45, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf151
    del primals_181
    buf153 = buf152[0]
    buf154 = buf152[1]
    del buf152
    buf155 = buf149; del buf149  # reuse
    buf156 = empty((256, ), device='cpu', dtype=torch.float32)
    buf157 = empty((256, ), device='cpu', dtype=torch.float32)
    buf158 = buf153; del buf153  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_21(c_void_p(buf158.data_ptr()), c_void_p(where_45.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_522.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()))
    del convolution_45
    del primals_91
    del squeeze_136
    del unsqueeze_522
    del where_45
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf159 = aten.convolution_backward(buf158, add_239, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_239
    del primals_180
    buf160 = buf159[0]
    buf161 = buf159[1]
    del buf159
    buf162 = buf102; del buf102  # reuse
    buf163 = buf156; del buf156  # reuse
    buf164 = empty((256, ), device='cpu', dtype=torch.float32)
    buf165 = empty((256, ), device='cpu', dtype=torch.float32)
    buf166 = buf158; del buf158  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_22(c_void_p(buf162.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(gt_89.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_534.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    del buf116
    del buf130
    del buf145
    del buf160
    del convolution_44
    del gt_89
    del primals_89
    del squeeze_133
    del unsqueeze_534
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf167 = aten.convolution_backward(buf166, where_43, primals_179, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf166
    del primals_179
    buf168 = buf167[0]
    buf169 = buf167[1]
    del buf167
    buf170 = buf164; del buf164  # reuse
    buf171 = empty((256, ), device='cpu', dtype=torch.float32)
    buf172 = empty((256, ), device='cpu', dtype=torch.float32)
    buf173 = buf168; del buf168  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_23(c_void_p(buf173.data_ptr()), c_void_p(where_43.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del convolution_43
    del primals_87
    del squeeze_130
    del unsqueeze_546
    del where_43
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf174 = aten.convolution_backward(buf173, add_228, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_228
    del primals_178
    buf175 = buf174[0]
    buf176 = buf174[1]
    del buf174
    buf177 = buf171; del buf171  # reuse
    buf178 = empty((256, ), device='cpu', dtype=torch.float32)
    buf179 = buf173; del buf173  # reuse
    buf180 = buf178; del buf178  # reuse
    cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_24(c_void_p(buf180.data_ptr()), c_void_p(gt_91.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_558.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()))
    del convolution_42
    del gt_91
    del primals_85
    del squeeze_127
    del unsqueeze_558
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf181 = aten.convolution_backward(buf179, where_41, primals_177, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf179
    del primals_177
    buf182 = buf181[0]
    buf183 = buf181[1]
    del buf181
    buf184 = empty((256, ), device='cpu', dtype=torch.float32)
    buf185 = empty((256, ), device='cpu', dtype=torch.float32)
    buf186 = empty((256, ), device='cpu', dtype=torch.float32)
    buf187 = buf182; del buf182  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_25(c_void_p(buf187.data_ptr()), c_void_p(where_41.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_570.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del convolution_41
    del primals_83
    del squeeze_124
    del unsqueeze_570
    del where_41
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf188 = aten.convolution_backward(buf187, add_217, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_217
    del primals_176
    buf189 = buf188[0]
    buf190 = buf188[1]
    del buf188
    buf191 = buf185; del buf185  # reuse
    buf192 = empty((256, ), device='cpu', dtype=torch.float32)
    buf193 = buf187; del buf187  # reuse
    buf195 = buf193; del buf193  # reuse
    buf194 = buf192; del buf192  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_26(c_void_p(buf195.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(gt_93.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf191.data_ptr()))
    del convolution_40
    del gt_93
    del primals_81
    del squeeze_121
    del unsqueeze_582
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf196 = aten.convolution_backward(buf195, where_39, primals_175, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf195
    del primals_175
    buf197 = buf196[0]
    buf198 = buf196[1]
    del buf196
    buf199 = empty((256, ), device='cpu', dtype=torch.float32)
    buf200 = empty((256, ), device='cpu', dtype=torch.float32)
    buf201 = empty((256, ), device='cpu', dtype=torch.float32)
    buf202 = buf197; del buf197  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_27(c_void_p(buf202.data_ptr()), c_void_p(where_39.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_594.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    del convolution_39
    del primals_79
    del squeeze_118
    del unsqueeze_594
    del where_39
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf203 = aten.convolution_backward(buf202, add_206, primals_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_206
    del primals_174
    buf204 = buf203[0]
    buf205 = buf203[1]
    del buf203
    buf206 = buf202; del buf202  # reuse
    buf207 = buf200; del buf200  # reuse
    buf208 = empty((256, ), device='cpu', dtype=torch.float32)
    buf209 = empty((256, ), device='cpu', dtype=torch.float32)
    buf210 = buf206; del buf206  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_28(c_void_p(buf210.data_ptr()), c_void_p(gt_95.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_606.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    del convolution_38
    del gt_95
    del primals_77
    del squeeze_115
    del unsqueeze_606
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf211 = aten.convolution_backward(buf210, where_37, primals_173, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf210
    del primals_173
    buf212 = buf211[0]
    buf213 = buf211[1]
    del buf211
    buf214 = buf208; del buf208  # reuse
    buf215 = empty((256, ), device='cpu', dtype=torch.float32)
    buf216 = empty((256, ), device='cpu', dtype=torch.float32)
    buf217 = buf212; del buf212  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_29(c_void_p(buf217.data_ptr()), c_void_p(where_37.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del convolution_37
    del primals_75
    del squeeze_112
    del unsqueeze_618
    del where_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf218 = aten.convolution_backward(buf217, getitem_95, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf217
    del getitem_95
    del primals_172
    buf219 = buf218[0]
    buf220 = buf218[1]
    del buf218
    buf221 = buf95; del buf95  # reuse
    buf222 = buf91; del buf91  # reuse
    buf223 = empty((512, ), device='cpu', dtype=torch.float32)
    buf224 = empty((512, ), device='cpu', dtype=torch.float32)
    buf225 = buf221; del buf221  # reuse
    cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_30(c_void_p(buf225.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(gt_97.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_630.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()))
    del buf162
    del buf175
    del buf189
    del buf204
    del buf219
    del convolution_36
    del gt_97
    del primals_73
    del squeeze_109
    del unsqueeze_630
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf226 = aten.convolution_backward(buf225, where_35, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf225
    del primals_171
    buf227 = buf226[0]
    buf228 = buf226[1]
    del buf226
    buf229 = buf223; del buf223  # reuse
    buf230 = empty((512, ), device='cpu', dtype=torch.float32)
    buf231 = empty((512, ), device='cpu', dtype=torch.float32)
    buf232 = buf227; del buf227  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_31(c_void_p(buf232.data_ptr()), c_void_p(where_35.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_642.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del buf230
    del convolution_35
    del primals_71
    del squeeze_106
    del unsqueeze_642
    del where_35
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf233 = aten.convolution_backward(buf232, where_34, primals_170, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_170
    buf234 = buf233[0]
    buf235 = buf233[1]
    del buf233
    buf236 = buf215; del buf215  # reuse
    buf237 = empty((256, ), device='cpu', dtype=torch.float32)
    buf238 = empty((256, ), device='cpu', dtype=torch.float32)
    buf239 = buf234; del buf234  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_32(c_void_p(buf239.data_ptr()), c_void_p(where_34.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del convolution_34
    del primals_69
    del squeeze_103
    del unsqueeze_654
    del where_34
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf240 = aten.convolution_backward(buf239, cat_2, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf239
    del cat_2
    del primals_169
    buf241 = buf240[0]
    buf242 = buf240[1]
    del buf240
    buf243 = empty((128, ), device='cpu', dtype=torch.float32)
    buf244 = empty((128, ), device='cpu', dtype=torch.float32)
    buf245 = empty((128, ), device='cpu', dtype=torch.float32)
    buf246 = reinterpret_tensor(buf232, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf232  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_33(c_void_p(gt_100.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_666.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()))
    del convolution_33
    del gt_100
    del primals_67
    del squeeze_100
    del unsqueeze_666
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf247 = aten.convolution_backward(buf246, add_175, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_175
    del primals_168
    buf248 = buf247[0]
    buf249 = buf247[1]
    del buf247
    buf250 = buf244; del buf244  # reuse
    buf251 = empty((128, ), device='cpu', dtype=torch.float32)
    buf252 = empty((128, ), device='cpu', dtype=torch.float32)
    buf253 = buf246; del buf246  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_34(c_void_p(gt_101.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_678.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del convolution_32
    del gt_101
    del primals_65
    del squeeze_97
    del unsqueeze_678
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf254 = aten.convolution_backward(buf253, where_31, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf253
    del primals_167
    buf255 = buf254[0]
    buf256 = buf254[1]
    del buf254
    buf257 = buf251; del buf251  # reuse
    buf258 = empty((128, ), device='cpu', dtype=torch.float32)
    buf259 = empty((128, ), device='cpu', dtype=torch.float32)
    buf260 = buf255; del buf255  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_35(c_void_p(buf260.data_ptr()), c_void_p(where_31.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    del convolution_31
    del primals_63
    del squeeze_94
    del unsqueeze_690
    del where_31
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf261 = aten.convolution_backward(buf260, add_164, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_164
    del primals_166
    buf262 = buf261[0]
    buf263 = buf261[1]
    del buf261
    buf264 = buf258; del buf258  # reuse
    buf265 = empty((128, ), device='cpu', dtype=torch.float32)
    buf266 = buf260; del buf260  # reuse
    buf267 = buf265; del buf265  # reuse
    cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_36(c_void_p(buf267.data_ptr()), c_void_p(gt_103.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_702.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    del convolution_30
    del gt_103
    del primals_61
    del squeeze_91
    del unsqueeze_702
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf268 = aten.convolution_backward(buf266, where_29, primals_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf266
    del primals_165
    buf269 = buf268[0]
    buf270 = buf268[1]
    del buf268
    buf271 = empty((128, ), device='cpu', dtype=torch.float32)
    buf272 = empty((128, ), device='cpu', dtype=torch.float32)
    buf273 = empty((128, ), device='cpu', dtype=torch.float32)
    buf274 = buf269; del buf269  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_37(c_void_p(buf274.data_ptr()), c_void_p(where_29.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_714.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()))
    del convolution_29
    del primals_59
    del squeeze_88
    del unsqueeze_714
    del where_29
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf275 = aten.convolution_backward(buf274, add_153, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_153
    del primals_164
    buf276 = buf275[0]
    buf277 = buf275[1]
    del buf275
    buf278 = buf272; del buf272  # reuse
    buf279 = empty((128, ), device='cpu', dtype=torch.float32)
    buf280 = buf274; del buf274  # reuse
    buf282 = buf280; del buf280  # reuse
    buf281 = buf279; del buf279  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_38(c_void_p(buf282.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(gt_105.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf278.data_ptr()))
    del convolution_28
    del gt_105
    del primals_57
    del squeeze_85
    del unsqueeze_726
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf283 = aten.convolution_backward(buf282, where_27, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf282
    del primals_163
    buf284 = buf283[0]
    buf285 = buf283[1]
    del buf283
    buf286 = empty((128, ), device='cpu', dtype=torch.float32)
    buf287 = empty((128, ), device='cpu', dtype=torch.float32)
    buf288 = empty((128, ), device='cpu', dtype=torch.float32)
    buf289 = buf284; del buf284  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_39(c_void_p(buf289.data_ptr()), c_void_p(where_27.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_738.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    del convolution_27
    del primals_55
    del squeeze_82
    del unsqueeze_738
    del where_27
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf290 = aten.convolution_backward(buf289, add_142, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_142
    del primals_162
    buf291 = buf290[0]
    buf292 = buf290[1]
    del buf290
    buf293 = buf289; del buf289  # reuse
    buf294 = buf287; del buf287  # reuse
    buf295 = empty((128, ), device='cpu', dtype=torch.float32)
    buf296 = empty((128, ), device='cpu', dtype=torch.float32)
    buf297 = buf293; del buf293  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_40(c_void_p(buf297.data_ptr()), c_void_p(gt_107.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_750.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    del convolution_26
    del gt_107
    del primals_53
    del squeeze_79
    del unsqueeze_750
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf298 = aten.convolution_backward(buf297, where_25, primals_161, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf297
    del primals_161
    buf299 = buf298[0]
    buf300 = buf298[1]
    del buf298
    buf301 = buf295; del buf295  # reuse
    buf302 = empty((128, ), device='cpu', dtype=torch.float32)
    buf303 = empty((128, ), device='cpu', dtype=torch.float32)
    buf304 = buf299; del buf299  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_41(c_void_p(buf304.data_ptr()), c_void_p(where_25.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_762.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()))
    del convolution_25
    del primals_51
    del squeeze_76
    del unsqueeze_762
    del where_25
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf305 = aten.convolution_backward(buf304, add_131, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_131
    del primals_160
    buf306 = buf305[0]
    buf307 = buf305[1]
    del buf305
    buf308 = buf248; del buf248  # reuse
    buf309 = buf302; del buf302  # reuse
    buf310 = empty((128, ), device='cpu', dtype=torch.float32)
    buf311 = empty((128, ), device='cpu', dtype=torch.float32)
    buf312 = buf304; del buf304  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_42(c_void_p(buf308.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(gt_109.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_774.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del buf262
    del buf276
    del buf291
    del buf306
    del convolution_24
    del gt_109
    del primals_49
    del squeeze_73
    del unsqueeze_774
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf313 = aten.convolution_backward(buf312, where_23, primals_159, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf312
    del primals_159
    buf314 = buf313[0]
    buf315 = buf313[1]
    del buf313
    buf316 = buf310; del buf310  # reuse
    buf317 = empty((128, ), device='cpu', dtype=torch.float32)
    buf318 = empty((128, ), device='cpu', dtype=torch.float32)
    buf319 = buf314; del buf314  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_43(c_void_p(buf319.data_ptr()), c_void_p(where_23.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_786.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()))
    del convolution_23
    del primals_47
    del squeeze_70
    del unsqueeze_786
    del where_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf320 = aten.convolution_backward(buf319, add_120, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_120
    del primals_158
    buf321 = buf320[0]
    buf322 = buf320[1]
    del buf320
    buf323 = buf317; del buf317  # reuse
    buf324 = empty((128, ), device='cpu', dtype=torch.float32)
    buf325 = buf319; del buf319  # reuse
    buf326 = buf324; del buf324  # reuse
    cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_44(c_void_p(buf326.data_ptr()), c_void_p(gt_111.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_798.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()))
    del convolution_22
    del gt_111
    del primals_45
    del squeeze_67
    del unsqueeze_798
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf327 = aten.convolution_backward(buf325, where_21, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf325
    del primals_157
    buf328 = buf327[0]
    buf329 = buf327[1]
    del buf327
    buf330 = empty((128, ), device='cpu', dtype=torch.float32)
    buf331 = empty((128, ), device='cpu', dtype=torch.float32)
    buf332 = empty((128, ), device='cpu', dtype=torch.float32)
    buf333 = buf328; del buf328  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_45(c_void_p(buf333.data_ptr()), c_void_p(where_21.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_810.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()))
    del convolution_21
    del primals_43
    del squeeze_64
    del unsqueeze_810
    del where_21
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf334 = aten.convolution_backward(buf333, add_109, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_109
    del primals_156
    buf335 = buf334[0]
    buf336 = buf334[1]
    del buf334
    buf337 = buf331; del buf331  # reuse
    buf338 = empty((128, ), device='cpu', dtype=torch.float32)
    buf339 = buf333; del buf333  # reuse
    buf341 = buf339; del buf339  # reuse
    buf340 = buf338; del buf338  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_46(c_void_p(buf341.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(gt_113.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_822.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf337.data_ptr()))
    del convolution_20
    del gt_113
    del primals_41
    del squeeze_61
    del unsqueeze_822
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf342 = aten.convolution_backward(buf341, where_19, primals_155, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf341
    del primals_155
    buf343 = buf342[0]
    buf344 = buf342[1]
    del buf342
    buf345 = empty((128, ), device='cpu', dtype=torch.float32)
    buf346 = empty((128, ), device='cpu', dtype=torch.float32)
    buf347 = empty((128, ), device='cpu', dtype=torch.float32)
    buf348 = buf343; del buf343  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_47(c_void_p(buf348.data_ptr()), c_void_p(where_19.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_834.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()))
    del convolution_19
    del primals_39
    del squeeze_58
    del unsqueeze_834
    del where_19
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf349 = aten.convolution_backward(buf348, add_98, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_98
    del primals_154
    buf350 = buf349[0]
    buf351 = buf349[1]
    del buf349
    buf352 = buf348; del buf348  # reuse
    buf353 = buf346; del buf346  # reuse
    buf354 = empty((128, ), device='cpu', dtype=torch.float32)
    buf355 = empty((128, ), device='cpu', dtype=torch.float32)
    buf356 = buf352; del buf352  # reuse
    cpp_fused_add_convolution_backward_leaky_relu_backward_native_batch_norm_backward_48(c_void_p(buf356.data_ptr()), c_void_p(gt_115.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_846.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    del convolution_18
    del gt_115
    del primals_37
    del squeeze_55
    del unsqueeze_846
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf357 = aten.convolution_backward(buf356, where_17, primals_153, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf356
    del primals_153
    buf358 = buf357[0]
    buf359 = buf357[1]
    del buf357
    buf360 = buf354; del buf354  # reuse
    buf361 = empty((128, ), device='cpu', dtype=torch.float32)
    buf362 = empty((128, ), device='cpu', dtype=torch.float32)
    buf363 = buf358; del buf358  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_49(c_void_p(buf363.data_ptr()), c_void_p(where_17.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_858.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    del convolution_17
    del primals_35
    del squeeze_52
    del unsqueeze_858
    del where_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf364 = aten.convolution_backward(buf363, getitem_49, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf363
    del getitem_49
    del primals_152
    buf365 = buf364[0]
    buf366 = buf364[1]
    del buf364
    buf367 = buf241; del buf241  # reuse
    buf368 = buf237; del buf237  # reuse
    buf369 = empty((256, ), device='cpu', dtype=torch.float32)
    buf370 = empty((256, ), device='cpu', dtype=torch.float32)
    buf371 = buf367; del buf367  # reuse
    cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_50(c_void_p(buf371.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(gt_117.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_870.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    del buf308
    del buf321
    del buf335
    del buf350
    del buf365
    del convolution_16
    del gt_117
    del primals_33
    del squeeze_49
    del unsqueeze_870
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf372 = aten.convolution_backward(buf371, where_15, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf371
    del primals_151
    buf373 = buf372[0]
    buf374 = buf372[1]
    del buf372
    buf375 = buf369; del buf369  # reuse
    buf376 = empty((256, ), device='cpu', dtype=torch.float32)
    buf377 = empty((256, ), device='cpu', dtype=torch.float32)
    buf378 = buf373; del buf373  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_51(c_void_p(buf378.data_ptr()), c_void_p(where_15.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_882.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del buf376
    del convolution_15
    del primals_31
    del squeeze_46
    del unsqueeze_882
    del where_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf379 = aten.convolution_backward(buf378, where_14, primals_150, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_150
    buf380 = buf379[0]
    buf381 = buf379[1]
    del buf379
    buf382 = buf361; del buf361  # reuse
    buf383 = empty((128, ), device='cpu', dtype=torch.float32)
    buf384 = empty((128, ), device='cpu', dtype=torch.float32)
    buf385 = buf380; del buf380  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_52(c_void_p(buf385.data_ptr()), c_void_p(where_14.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_894.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()))
    del convolution_14
    del primals_29
    del squeeze_43
    del unsqueeze_894
    del where_14
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf386 = aten.convolution_backward(buf385, cat_1, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf385
    del cat_1
    del primals_149
    buf387 = buf386[0]
    buf388 = buf386[1]
    del buf386
    buf389 = empty((64, ), device='cpu', dtype=torch.float32)
    buf390 = empty((64, ), device='cpu', dtype=torch.float32)
    buf391 = empty((64, ), device='cpu', dtype=torch.float32)
    buf392 = reinterpret_tensor(buf378, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf378  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_53(c_void_p(gt_120.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_906.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()))
    del convolution_13
    del gt_120
    del primals_27
    del squeeze_40
    del unsqueeze_906
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf393 = aten.convolution_backward(buf392, add_67, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_67
    del primals_148
    buf394 = buf393[0]
    buf395 = buf393[1]
    del buf393
    buf396 = buf390; del buf390  # reuse
    buf397 = empty((64, ), device='cpu', dtype=torch.float32)
    buf398 = empty((64, ), device='cpu', dtype=torch.float32)
    buf399 = buf392; del buf392  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_54(c_void_p(gt_121.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_918.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()))
    del convolution_12
    del gt_121
    del primals_25
    del squeeze_37
    del unsqueeze_918
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf400 = aten.convolution_backward(buf399, where_11, primals_147, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf399
    del primals_147
    buf401 = buf400[0]
    buf402 = buf400[1]
    del buf400
    buf403 = buf397; del buf397  # reuse
    buf404 = empty((64, ), device='cpu', dtype=torch.float32)
    buf405 = empty((64, ), device='cpu', dtype=torch.float32)
    buf406 = buf401; del buf401  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_55(c_void_p(buf406.data_ptr()), c_void_p(where_11.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_930.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()))
    del convolution_11
    del primals_23
    del squeeze_34
    del unsqueeze_930
    del where_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf407 = aten.convolution_backward(buf406, add_56, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_56
    del primals_146
    buf408 = buf407[0]
    buf409 = buf407[1]
    del buf407
    buf410 = buf404; del buf404  # reuse
    buf411 = empty((64, ), device='cpu', dtype=torch.float32)
    buf412 = buf406; del buf406  # reuse
    buf413 = buf411; del buf411  # reuse
    cpp_fused_add_leaky_relu_backward_native_batch_norm_backward_56(c_void_p(buf413.data_ptr()), c_void_p(gt_123.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_942.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf412.data_ptr()))
    del convolution_10
    del gt_123
    del primals_21
    del squeeze_31
    del unsqueeze_942
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf414 = aten.convolution_backward(buf412, where_9, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf412
    del primals_145
    buf415 = buf414[0]
    buf416 = buf414[1]
    del buf414
    buf417 = empty((64, ), device='cpu', dtype=torch.float32)
    buf418 = empty((64, ), device='cpu', dtype=torch.float32)
    buf419 = empty((64, ), device='cpu', dtype=torch.float32)
    buf420 = buf415; del buf415  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_57(c_void_p(buf420.data_ptr()), c_void_p(where_9.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_954.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()))
    del convolution_9
    del primals_19
    del squeeze_28
    del unsqueeze_954
    del where_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf421 = aten.convolution_backward(buf420, getitem_27, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf420
    del getitem_27
    del primals_144
    buf422 = buf421[0]
    buf423 = buf421[1]
    del buf421
    buf424 = buf387; del buf387  # reuse
    buf425 = buf383; del buf383  # reuse
    buf426 = empty((128, ), device='cpu', dtype=torch.float32)
    buf427 = empty((128, ), device='cpu', dtype=torch.float32)
    buf428 = buf424; del buf424  # reuse
    cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_58(c_void_p(buf428.data_ptr()), c_void_p(gt_125.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_966.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()))
    del buf394
    del buf408
    del buf422
    del convolution_8
    del gt_125
    del primals_17
    del squeeze_25
    del unsqueeze_966
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf429 = aten.convolution_backward(buf428, where_7, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf428
    del primals_143
    buf430 = buf429[0]
    buf431 = buf429[1]
    del buf429
    buf432 = buf426; del buf426  # reuse
    buf433 = empty((128, ), device='cpu', dtype=torch.float32)
    buf434 = empty((128, ), device='cpu', dtype=torch.float32)
    buf435 = buf430; del buf430  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_59(c_void_p(buf435.data_ptr()), c_void_p(where_7.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_978.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()))
    del convolution_7
    del primals_15
    del squeeze_22
    del unsqueeze_978
    del where_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf436 = aten.convolution_backward(buf435, where_6, primals_142, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf435
    del primals_142
    buf437 = buf436[0]
    buf438 = buf436[1]
    del buf436
    buf439 = buf418; del buf418  # reuse
    buf440 = empty((64, ), device='cpu', dtype=torch.float32)
    buf441 = empty((64, ), device='cpu', dtype=torch.float32)
    buf442 = buf437; del buf437  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_60(c_void_p(buf442.data_ptr()), c_void_p(where_6.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_990.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()))
    del convolution_6
    del primals_13
    del squeeze_19
    del unsqueeze_990
    del where_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf443 = aten.convolution_backward(buf442, cat, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat
    del primals_141
    buf444 = buf443[0]
    buf445 = buf443[1]
    del buf443
    buf446 = buf440; del buf440  # reuse
    buf447 = empty((64, ), device='cpu', dtype=torch.float32)
    buf448 = empty((64, ), device='cpu', dtype=torch.float32)
    buf449 = buf442; del buf442  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_61(c_void_p(gt_128.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_1002.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()))
    del convolution_5
    del gt_128
    del primals_11
    del squeeze_16
    del unsqueeze_1002
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf450 = aten.convolution_backward(buf449, add_25, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_25
    del primals_140
    buf451 = buf450[0]
    buf452 = buf450[1]
    del buf450
    buf453 = buf447; del buf447  # reuse
    buf454 = empty((64, ), device='cpu', dtype=torch.float32)
    buf455 = empty((64, ), device='cpu', dtype=torch.float32)
    buf456 = buf449; del buf449  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_62(c_void_p(gt_129.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_1014.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()))
    del convolution_4
    del gt_129
    del primals_9
    del squeeze_13
    del unsqueeze_1014
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf457 = aten.convolution_backward(buf456, where_3, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf456
    del primals_139
    buf458 = buf457[0]
    buf459 = buf457[1]
    del buf457
    buf460 = empty((32, ), device='cpu', dtype=torch.float32)
    buf461 = empty((32, ), device='cpu', dtype=torch.float32)
    buf462 = empty((32, ), device='cpu', dtype=torch.float32)
    buf463 = buf458; del buf458  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_63(c_void_p(buf463.data_ptr()), c_void_p(where_3.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_1026.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()))
    del convolution_3
    del primals_7
    del squeeze_10
    del unsqueeze_1026
    del where_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf464 = aten.convolution_backward(buf463, getitem_9, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf463
    del getitem_9
    del primals_138
    buf465 = buf464[0]
    buf466 = buf464[1]
    del buf464
    buf467 = buf433; del buf433  # reuse
    buf468 = empty((128, ), device='cpu', dtype=torch.float32)
    buf469 = buf444; del buf444  # reuse
    buf470 = buf468; del buf468  # reuse
    buf471 = buf469; del buf469  # reuse
    cpp_fused_cat_convolution_backward_leaky_relu_backward_native_batch_norm_backward_64(c_void_p(buf471.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(gt_131.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_1038.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf467.data_ptr()))
    del buf451
    del buf465
    del convolution_2
    del gt_131
    del primals_5
    del squeeze_7
    del unsqueeze_1038
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf472 = aten.convolution_backward(buf471, where_1, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf471
    del primals_137
    buf473 = buf472[0]
    buf474 = buf472[1]
    del buf472
    buf475 = buf454; del buf454  # reuse
    buf476 = empty((64, ), device='cpu', dtype=torch.float32)
    buf477 = empty((64, ), device='cpu', dtype=torch.float32)
    buf478 = buf473; del buf473  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_65(c_void_p(buf478.data_ptr()), c_void_p(where_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_1050.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    del buf476
    del convolution_1
    del primals_3
    del squeeze_4
    del unsqueeze_1050
    del where_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf479 = aten.convolution_backward(buf478, where, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf478
    del primals_136
    buf480 = buf479[0]
    buf481 = buf479[1]
    del buf479
    buf482 = buf461; del buf461  # reuse
    buf483 = empty((32, ), device='cpu', dtype=torch.float32)
    buf484 = empty((32, ), device='cpu', dtype=torch.float32)
    buf485 = buf480; del buf480  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_66(c_void_p(buf485.data_ptr()), c_void_p(where.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_1062.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()))
    del buf483
    del convolution
    del primals_1
    del squeeze_1
    del unsqueeze_1062
    del where
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf486 = aten.convolution_backward(buf485, primals_405, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf485
    del primals_135
    del primals_405
    buf487 = buf486[1]
    return (buf484, buf482, buf477, buf475, buf470, buf467, buf462, buf460, buf455, buf453, buf448, buf446, buf441, buf439, buf434, buf432, buf427, buf425, buf419, buf417, buf413, buf410, buf405, buf403, buf398, buf396, buf391, buf389, buf384, buf382, buf377, buf375, buf370, buf368, buf362, buf360, buf355, buf353, buf347, buf345, buf340, buf337, buf332, buf330, buf326, buf323, buf318, buf316, buf311, buf309, buf303, buf301, buf296, buf294, buf288, buf286, buf281, buf278, buf273, buf271, buf267, buf264, buf259, buf257, buf252, buf250, buf245, buf243, buf238, buf236, buf231, buf229, buf224, buf222, buf216, buf214, buf209, buf207, buf201, buf199, buf194, buf191, buf186, buf184, buf180, buf177, buf172, buf170, buf165, buf163, buf157, buf155, buf150, buf148, buf142, buf140, buf135, buf132, buf127, buf125, buf121, buf118, buf113, buf111, buf106, buf104, buf99, buf97, buf92, buf90, buf85, buf83, buf78, buf76, buf70, buf68, buf63, buf61, buf55, buf53, buf48, buf45, buf40, buf38, buf34, buf31, buf26, buf24, buf19, buf17, buf12, buf10, buf5, buf3, buf487, buf481, buf474, buf466, buf459, buf452, buf445, buf438, buf431, buf423, buf416, buf409, buf402, buf395, buf388, buf381, buf374, buf366, buf359, buf351, buf344, buf336, buf329, buf322, buf315, buf307, buf300, buf292, buf285, buf277, buf270, buf263, buf256, buf249, buf242, buf235, buf228, buf220, buf213, buf205, buf198, buf190, buf183, buf176, buf169, buf161, buf154, buf146, buf139, buf131, buf124, buf117, buf110, buf103, buf96, buf89, buf82, buf74, buf67, buf59, buf52, buf44, buf37, buf30, buf23, buf16, buf9, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((1024, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_405 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 256, 256), (2097152, 1, 8192, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    where = rand_strided((8, 32, 256, 256), (2097152, 1, 8192, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    where_1 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_9 = rand_strided((8, 64, 128, 128), (2097152, 16384, 128, 1), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    where_3 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_25 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    where_6 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_7 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((8, 64, 64, 64), (524288, 4096, 64, 1), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    where_9 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_56 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    where_11 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_67 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_14 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_15 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_49 = rand_strided((8, 128, 32, 32), (262144, 1024, 32, 1), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_17 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_98 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_19 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_109 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_21 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_120 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_23 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_131 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_25 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_142 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_27 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_153 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_29 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_164 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    where_31 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_175 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_34 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    where_35 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_95 = rand_strided((8, 256, 16, 16), (131072, 256, 16, 1), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_37 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_206 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_39 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_217 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_41 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_228 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_43 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_239 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_45 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_250 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_47 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_261 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_49 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_272 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    where_51 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_283 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    where_54 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    where_55 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_141 = rand_strided((8, 512, 8, 8), (65536, 64, 8, 1), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    where_57 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    add_314 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    where_59 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    add_325 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    where_61 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_187 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    add_336 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_190 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    where_63 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_193 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    add_347 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_196 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cpu', dtype=torch.float32)
    squeeze_199 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    gt_67 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cpu', dtype=torch.bool)
    unsqueeze_270 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_68 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.bool)
    unsqueeze_282 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_69 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.bool)
    unsqueeze_294 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_71 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.bool)
    unsqueeze_318 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_73 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.bool)
    unsqueeze_342 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_75 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.bool)
    unsqueeze_366 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_77 = rand_strided((8, 1024, 8, 8), (65536, 1, 8192, 1024), device='cpu', dtype=torch.bool)
    unsqueeze_390 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_80 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.bool)
    unsqueeze_426 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_81 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.bool)
    unsqueeze_438 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_83 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.bool)
    unsqueeze_462 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_85 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.bool)
    unsqueeze_486 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_87 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.bool)
    unsqueeze_510 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_89 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.bool)
    unsqueeze_534 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_91 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_93 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.bool)
    unsqueeze_582 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_95 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.bool)
    unsqueeze_606 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_97 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.bool)
    unsqueeze_630 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_100 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.bool)
    unsqueeze_666 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_101 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.bool)
    unsqueeze_678 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_103 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.bool)
    unsqueeze_702 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_105 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.bool)
    unsqueeze_726 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_107 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.bool)
    unsqueeze_750 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_109 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.bool)
    unsqueeze_774 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_786 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_111 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.bool)
    unsqueeze_798 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_113 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.bool)
    unsqueeze_822 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_115 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.bool)
    unsqueeze_846 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_858 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_117 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.bool)
    unsqueeze_870 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_882 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_894 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_120 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.bool)
    unsqueeze_906 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_121 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.bool)
    unsqueeze_918 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_930 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_123 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.bool)
    unsqueeze_942 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_954 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_125 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.bool)
    unsqueeze_966 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_978 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_990 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_128 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1002 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_129 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1014 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1026 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    gt_131 = rand_strided((8, 128, 128, 128), (2097152, 1, 16384, 128), device='cpu', dtype=torch.bool)
    unsqueeze_1038 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1050 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1062 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_405, convolution, squeeze_1, where, convolution_1, squeeze_4, where_1, convolution_2, squeeze_7, getitem_9, convolution_3, squeeze_10, where_3, convolution_4, squeeze_13, add_25, convolution_5, squeeze_16, cat, convolution_6, squeeze_19, where_6, convolution_7, squeeze_22, where_7, convolution_8, squeeze_25, getitem_27, convolution_9, squeeze_28, where_9, convolution_10, squeeze_31, add_56, convolution_11, squeeze_34, where_11, convolution_12, squeeze_37, add_67, convolution_13, squeeze_40, cat_1, convolution_14, squeeze_43, where_14, convolution_15, squeeze_46, where_15, convolution_16, squeeze_49, getitem_49, convolution_17, squeeze_52, where_17, convolution_18, squeeze_55, add_98, convolution_19, squeeze_58, where_19, convolution_20, squeeze_61, add_109, convolution_21, squeeze_64, where_21, convolution_22, squeeze_67, add_120, convolution_23, squeeze_70, where_23, convolution_24, squeeze_73, add_131, convolution_25, squeeze_76, where_25, convolution_26, squeeze_79, add_142, convolution_27, squeeze_82, where_27, convolution_28, squeeze_85, add_153, convolution_29, squeeze_88, where_29, convolution_30, squeeze_91, add_164, convolution_31, squeeze_94, where_31, convolution_32, squeeze_97, add_175, convolution_33, squeeze_100, cat_2, convolution_34, squeeze_103, where_34, convolution_35, squeeze_106, where_35, convolution_36, squeeze_109, getitem_95, convolution_37, squeeze_112, where_37, convolution_38, squeeze_115, add_206, convolution_39, squeeze_118, where_39, convolution_40, squeeze_121, add_217, convolution_41, squeeze_124, where_41, convolution_42, squeeze_127, add_228, convolution_43, squeeze_130, where_43, convolution_44, squeeze_133, add_239, convolution_45, squeeze_136, where_45, convolution_46, squeeze_139, add_250, convolution_47, squeeze_142, where_47, convolution_48, squeeze_145, add_261, convolution_49, squeeze_148, where_49, convolution_50, squeeze_151, add_272, convolution_51, squeeze_154, where_51, convolution_52, squeeze_157, add_283, convolution_53, squeeze_160, cat_3, convolution_54, squeeze_163, where_54, convolution_55, squeeze_166, where_55, convolution_56, squeeze_169, getitem_141, convolution_57, squeeze_172, where_57, convolution_58, squeeze_175, add_314, convolution_59, squeeze_178, where_59, convolution_60, squeeze_181, add_325, convolution_61, squeeze_184, where_61, convolution_62, squeeze_187, add_336, convolution_63, squeeze_190, where_63, convolution_64, squeeze_193, add_347, convolution_65, squeeze_196, cat_4, convolution_66, squeeze_199, clone, permute_1, gt_67, unsqueeze_270, gt_68, unsqueeze_282, gt_69, unsqueeze_294, unsqueeze_306, gt_71, unsqueeze_318, unsqueeze_330, gt_73, unsqueeze_342, unsqueeze_354, gt_75, unsqueeze_366, unsqueeze_378, gt_77, unsqueeze_390, unsqueeze_402, unsqueeze_414, gt_80, unsqueeze_426, gt_81, unsqueeze_438, unsqueeze_450, gt_83, unsqueeze_462, unsqueeze_474, gt_85, unsqueeze_486, unsqueeze_498, gt_87, unsqueeze_510, unsqueeze_522, gt_89, unsqueeze_534, unsqueeze_546, gt_91, unsqueeze_558, unsqueeze_570, gt_93, unsqueeze_582, unsqueeze_594, gt_95, unsqueeze_606, unsqueeze_618, gt_97, unsqueeze_630, unsqueeze_642, unsqueeze_654, gt_100, unsqueeze_666, gt_101, unsqueeze_678, unsqueeze_690, gt_103, unsqueeze_702, unsqueeze_714, gt_105, unsqueeze_726, unsqueeze_738, gt_107, unsqueeze_750, unsqueeze_762, gt_109, unsqueeze_774, unsqueeze_786, gt_111, unsqueeze_798, unsqueeze_810, gt_113, unsqueeze_822, unsqueeze_834, gt_115, unsqueeze_846, unsqueeze_858, gt_117, unsqueeze_870, unsqueeze_882, unsqueeze_894, gt_120, unsqueeze_906, gt_121, unsqueeze_918, unsqueeze_930, gt_123, unsqueeze_942, unsqueeze_954, gt_125, unsqueeze_966, unsqueeze_978, unsqueeze_990, gt_128, unsqueeze_1002, gt_129, unsqueeze_1014, unsqueeze_1026, gt_131, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('cspdarknet53', benchmark_compiled_module)
