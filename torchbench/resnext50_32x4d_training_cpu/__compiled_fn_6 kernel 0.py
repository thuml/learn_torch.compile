
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
extern "C" void kernel(const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
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
                        tmp15.store(out_ptr1 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(49.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(0.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                            auto tmp10 = tmp8 - tmp9;
                            auto tmp11 = tmp7 * tmp10;
                            auto tmp13 = to_float_mask(tmp12 <= tmp6);
                            auto tmp15 = tmp7 + tmp14;
                            auto tmp16 = decltype(tmp6)::blendv(tmp15, tmp6, tmp13);
                            auto tmp19 = tmp17 - tmp18;
                            auto tmp20 = tmp16 * tmp19;
                            tmp_acc0_vec = tmp_acc0_vec + tmp7;
                            tmp_acc1_vec = tmp_acc1_vec + tmp11;
                            tmp_acc2_vec = tmp_acc2_vec + tmp16;
                            tmp_acc3_vec = tmp_acc3_vec + tmp20;
                        }
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
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


cpp_fused_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp4 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = static_cast<float>(49.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 / tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp4);
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp14 = static_cast<float>(1e-05);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 + tmp15;
                        auto tmp17 = tmp16.rsqrt();
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp20 = tmp12 * tmp19;
                        tmp20.store(out_ptr0 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp6 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
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
                        tmp17.store(in_out_ptr0 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                    }
                }
            }
        }
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (2048L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (2048L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    tmp14.store(out_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_19 = async_compile.cpp('''
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_28 = async_compile.cpp('''
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    tmp14.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.cpp('''
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_47 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    tmp14.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_60 = async_compile.cpp('''
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc4 = 0;
                    at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_63 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp21.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68 = async_compile.cpp('''
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
                    auto tmp7 = static_cast<float>(1e-05);
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, convolution, relu, getitem, getitem_1, convolution_1, relu_1, convolution_2, relu_2, convolution_3, convolution_4, relu_3, convolution_5, relu_4, convolution_6, relu_5, convolution_7, relu_6, convolution_8, relu_7, convolution_9, relu_8, convolution_10, relu_9, convolution_11, relu_10, convolution_12, relu_11, convolution_13, convolution_14, relu_12, convolution_15, relu_13, convolution_16, relu_14, convolution_17, relu_15, convolution_18, relu_16, convolution_19, relu_17, convolution_20, relu_18, convolution_21, relu_19, convolution_22, relu_20, convolution_23, relu_21, convolution_24, relu_22, convolution_25, relu_23, convolution_26, convolution_27, relu_24, convolution_28, relu_25, convolution_29, relu_26, convolution_30, relu_27, convolution_31, relu_28, convolution_32, relu_29, convolution_33, relu_30, convolution_34, relu_31, convolution_35, relu_32, convolution_36, relu_33, convolution_37, relu_34, convolution_38, relu_35, convolution_39, relu_36, convolution_40, relu_37, convolution_41, relu_38, convolution_42, relu_39, convolution_43, relu_40, convolution_44, relu_41, convolution_45, convolution_46, relu_42, convolution_47, relu_43, convolution_48, relu_44, convolution_49, relu_45, convolution_50, relu_46, convolution_51, relu_47, convolution_52, view, permute_1, le, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (128, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_10, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_16, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_19, (128, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_22, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_28, (128, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_34, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (256, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_40, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_46, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (256, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_52, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_58, (256, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_61, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_64, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_67, (256, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_70, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_73, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_76, (512, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_77, (512, ), (1, ))
    assert_size_stride(primals_79, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_85, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_88, (512, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_91, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_94, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_97, (512, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_100, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_103, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_106, (512, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_109, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_112, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_115, (512, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_118, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_121, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_124, (512, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_127, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_134, (1024, ), (1, ))
    assert_size_stride(primals_136, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_142, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_143, (1024, ), (1, ))
    assert_size_stride(primals_145, (1024, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_148, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_151, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_152, (1024, ), (1, ))
    assert_size_stride(primals_154, (1024, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_155, (1024, ), (1, ))
    assert_size_stride(primals_157, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_162, (64, ), (1, ))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_165, (128, ), (1, ))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_189, (128, ), (1, ))
    assert_size_stride(primals_190, (128, ), (1, ))
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, ), (1, ))
    assert_size_stride(primals_201, (512, ), (1, ))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_207, (256, ), (1, ))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_225, (256, ), (1, ))
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_228, (256, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (512, ), (1, ))
    assert_size_stride(primals_237, (512, ), (1, ))
    assert_size_stride(primals_238, (512, ), (1, ))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (512, ), (1, ))
    assert_size_stride(primals_249, (512, ), (1, ))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_252, (1024, ), (1, ))
    assert_size_stride(primals_253, (1024, ), (1, ))
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (512, ), (1, ))
    assert_size_stride(primals_261, (1024, ), (1, ))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (512, ), (1, ))
    assert_size_stride(primals_267, (512, ), (1, ))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_273, (512, ), (1, ))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (512, ), (1, ))
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_282, (512, ), (1, ))
    assert_size_stride(primals_283, (512, ), (1, ))
    assert_size_stride(primals_285, (512, ), (1, ))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, ), (1, ))
    assert_size_stride(primals_291, (1024, ), (1, ))
    assert_size_stride(primals_292, (1024, ), (1, ))
    assert_size_stride(primals_294, (1024, ), (1, ))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_297, (2048, ), (1, ))
    assert_size_stride(primals_298, (2048, ), (1, ))
    assert_size_stride(primals_300, (2048, ), (1, ))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (1024, ), (1, ))
    assert_size_stride(primals_306, (1024, ), (1, ))
    assert_size_stride(primals_307, (1024, ), (1, ))
    assert_size_stride(primals_309, (2048, ), (1, ))
    assert_size_stride(primals_310, (2048, ), (1, ))
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, ), (1, ))
    assert_size_stride(primals_315, (1024, ), (1, ))
    assert_size_stride(primals_316, (1024, ), (1, ))
    assert_size_stride(primals_318, (2048, ), (1, ))
    assert_size_stride(primals_319, (2048, ), (1, ))
    assert_size_stride(primals_321, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(relu, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(getitem_1, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_1, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_1, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_2, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_2, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_3, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_4, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_3, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_5, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_4, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_6, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_5, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_7, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_6, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_8, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_7, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_9, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_8, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_10, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_9, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_11, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_10, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_12, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_11, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_13, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_14, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_12, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_15, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_13, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_16, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_14, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_17, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_15, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_18, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_16, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_19, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_17, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_20, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_18, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_21, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_19, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_22, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(relu_20, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_23, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_21, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_24, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_22, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_25, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_23, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_26, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_27, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(relu_24, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_28, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_25, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_29, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_26, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_30, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(relu_27, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_31, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_28, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_32, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_29, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_33, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(relu_30, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_34, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_31, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_35, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_32, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_36, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(relu_33, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_37, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_34, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_38, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_35, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_39, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(relu_36, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_40, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_37, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_41, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(relu_38, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_42, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(relu_39, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_43, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(relu_40, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_44, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(relu_41, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_45, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_46, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(relu_42, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_47, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(relu_43, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_48, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(relu_44, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_49, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(relu_45, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_50, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(relu_46, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_51, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(relu_47, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_52, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(view, (4, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), view, out=buf1)
    del view
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_view_0(c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf6.data_ptr()))
    del primals_158
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_157
    buf8 = buf7[0]
    buf13 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1(c_void_p(relu_47.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf13.data_ptr()))
    del primals_155
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf14 = aten.convolution_backward(buf13, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del primals_154
    buf15 = buf14[0]
    buf20 = buf13; del buf13  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2(c_void_p(relu_46.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf20.data_ptr()))
    del primals_152
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf21 = aten.convolution_backward(buf20, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf20
    del primals_151
    buf22 = buf21[0]
    buf3 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf4 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf24 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf25 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf5 = buf4; del buf4  # reuse
    cpp_fused_add_div_native_batch_norm_backward_threshold_backward_3(c_void_p(buf5.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del convolution_49
    del convolution_52
    del primals_309
    del primals_318
    del primals_319
    buf9 = buf7[1]
    del buf7
    buf10 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf11 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf12 = buf11; del buf11  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_4(c_void_p(buf12.data_ptr()), c_void_p(relu_47.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf8
    del convolution_51
    del primals_315
    del primals_316
    del relu_47
    buf16 = buf14[1]
    del buf14
    buf17 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf18 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf19 = buf18; del buf18  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_5(c_void_p(buf19.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(buf17.data_ptr()))
    del buf15
    del convolution_50
    del primals_312
    del primals_313
    del relu_46
    buf23 = buf21[1]
    del buf21
    buf26 = buf25; del buf25  # reuse
    buf27 = buf6; del buf6  # reuse
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6(c_void_p(buf26.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf27.data_ptr()))
    del primals_149
    del primals_310
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf28 = aten.convolution_backward(buf27, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_148
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf31 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf32 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf33 = buf32; del buf32  # reuse
    buf34 = buf29; del buf29  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7(c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(relu_44.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf31.data_ptr()))
    del convolution_48
    del primals_146
    del primals_306
    del primals_307
    del relu_44
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf35 = aten.convolution_backward(buf34, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf34
    del primals_145
    buf36 = buf35[0]
    buf37 = buf35[1]
    del buf35
    buf38 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf39 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf40 = buf39; del buf39  # reuse
    buf41 = buf36; del buf36  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(relu_43.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf38.data_ptr()))
    del convolution_47
    del primals_143
    del primals_303
    del primals_304
    del relu_43
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf42 = aten.convolution_backward(buf41, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf41
    del primals_142
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    buf45 = buf22; del buf22  # reuse
    buf46 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf47 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf53 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf48 = buf47; del buf47  # reuse
    buf49 = buf27; del buf27  # reuse
    buf55 = empty_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_9(c_void_p(buf45.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf55.data_ptr()))
    del buf0
    del buf43
    del buf45
    del convolution_45
    del convolution_46
    del le
    del primals_137
    del primals_140
    del primals_297
    del primals_300
    del primals_301
    del relu_42
    del relu_45
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf50 = aten.convolution_backward(buf49, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf49
    del primals_139
    buf51 = buf50[0]
    buf52 = buf50[1]
    del buf50
    buf54 = buf53; del buf53  # reuse
    cpp_fused_native_batch_norm_backward_10(c_void_p(buf54.data_ptr()), c_void_p(primals_298.data_ptr()))
    del primals_298
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf56 = aten.convolution_backward(buf55, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf55
    del primals_136
    buf57 = buf56[0]
    buf58 = buf56[1]
    del buf56
    buf59 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf60 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf61 = buf60; del buf60  # reuse
    buf62 = buf57; del buf57  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf59.data_ptr()))
    del convolution_44
    del primals_134
    del primals_294
    del primals_295
    del relu_41
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf63 = aten.convolution_backward(buf62, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf62
    del primals_133
    buf64 = buf63[0]
    buf65 = buf63[1]
    del buf63
    buf66 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf67 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf68 = buf67; del buf67  # reuse
    buf69 = buf64; del buf64  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf66.data_ptr()))
    del convolution_43
    del primals_131
    del primals_291
    del primals_292
    del relu_40
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf70 = aten.convolution_backward(buf69, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_130
    buf71 = buf70[0]
    buf72 = buf70[1]
    del buf70
    buf73 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf74 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf75 = buf74; del buf74  # reuse
    buf76 = buf69; del buf69  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(buf75.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf76.data_ptr()))
    del convolution_42
    del primals_128
    del primals_288
    del primals_289
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf77 = aten.convolution_backward(buf76, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_127
    buf78 = buf77[0]
    buf79 = buf77[1]
    del buf77
    buf80 = empty((512, ), device='cpu', dtype=torch.float32)
    buf81 = empty((512, ), device='cpu', dtype=torch.float32)
    buf82 = buf81; del buf81  # reuse
    buf83 = buf78; del buf78  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf80.data_ptr()))
    del convolution_41
    del primals_125
    del primals_285
    del primals_286
    del relu_38
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf84 = aten.convolution_backward(buf83, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf83
    del primals_124
    buf85 = buf84[0]
    buf86 = buf84[1]
    del buf84
    buf87 = empty((512, ), device='cpu', dtype=torch.float32)
    buf88 = empty((512, ), device='cpu', dtype=torch.float32)
    buf89 = buf88; del buf88  # reuse
    buf90 = buf85; del buf85  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf87.data_ptr()))
    del convolution_40
    del primals_122
    del primals_282
    del primals_283
    del relu_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf91 = aten.convolution_backward(buf90, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_121
    buf92 = buf91[0]
    buf93 = buf91[1]
    del buf91
    buf94 = buf51; del buf51  # reuse
    buf98 = buf76; del buf76  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_16(c_void_p(buf94.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf98.data_ptr()))
    del buf71
    del buf92
    del primals_119
    del relu_36
    del relu_39
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf99 = aten.convolution_backward(buf98, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_118
    buf100 = buf99[0]
    buf105 = buf90; del buf90  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(relu_35.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf105.data_ptr()))
    del primals_116
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf106 = aten.convolution_backward(buf105, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del primals_115
    buf107 = buf106[0]
    buf112 = buf105; del buf105  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(relu_34.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf112.data_ptr()))
    del primals_113
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf113 = aten.convolution_backward(buf112, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf112
    del primals_112
    buf114 = buf113[0]
    buf95 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf96 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf116 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf117 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf97 = buf96; del buf96  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_19(c_void_p(buf97.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del convolution_36
    del convolution_39
    del primals_270
    del primals_279
    del primals_280
    buf101 = buf99[1]
    del buf99
    buf102 = empty((512, ), device='cpu', dtype=torch.float32)
    buf103 = empty((512, ), device='cpu', dtype=torch.float32)
    buf104 = buf103; del buf103  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_20(c_void_p(buf104.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(buf102.data_ptr()))
    del buf100
    del convolution_38
    del primals_276
    del primals_277
    del relu_35
    buf108 = buf106[1]
    del buf106
    buf109 = empty((512, ), device='cpu', dtype=torch.float32)
    buf110 = empty((512, ), device='cpu', dtype=torch.float32)
    buf111 = buf110; del buf110  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_21(c_void_p(buf111.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(buf109.data_ptr()))
    del buf107
    del convolution_37
    del primals_273
    del primals_274
    del relu_34
    buf115 = buf113[1]
    del buf113
    buf118 = buf117; del buf117  # reuse
    buf119 = buf98; del buf98  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(buf118.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf119.data_ptr()))
    del primals_110
    del primals_271
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf120 = aten.convolution_backward(buf119, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_109
    buf121 = buf120[0]
    buf122 = buf120[1]
    del buf120
    buf123 = empty((512, ), device='cpu', dtype=torch.float32)
    buf124 = empty((512, ), device='cpu', dtype=torch.float32)
    buf125 = buf124; del buf124  # reuse
    buf126 = buf121; del buf121  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf123.data_ptr()))
    del convolution_35
    del primals_107
    del primals_267
    del primals_268
    del relu_32
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf127 = aten.convolution_backward(buf126, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf126
    del primals_106
    buf128 = buf127[0]
    buf129 = buf127[1]
    del buf127
    buf130 = empty((512, ), device='cpu', dtype=torch.float32)
    buf131 = empty((512, ), device='cpu', dtype=torch.float32)
    buf132 = buf131; del buf131  # reuse
    buf133 = buf128; del buf128  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(relu_31.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf130.data_ptr()))
    del convolution_34
    del primals_104
    del primals_264
    del primals_265
    del relu_31
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf134 = aten.convolution_backward(buf133, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_103
    buf135 = buf134[0]
    buf136 = buf134[1]
    del buf134
    buf137 = buf114; del buf114  # reuse
    buf141 = buf119; del buf119  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_25(c_void_p(buf137.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf141.data_ptr()))
    del buf135
    del primals_101
    del relu_30
    del relu_33
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf142 = aten.convolution_backward(buf141, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_100
    buf143 = buf142[0]
    buf148 = buf133; del buf133  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26(c_void_p(relu_29.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf148.data_ptr()))
    del primals_98
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf149 = aten.convolution_backward(buf148, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del primals_97
    buf150 = buf149[0]
    buf155 = buf148; del buf148  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(relu_28.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf155.data_ptr()))
    del primals_95
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf156 = aten.convolution_backward(buf155, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf155
    del primals_94
    buf157 = buf156[0]
    buf138 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf139 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf159 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf160 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf140 = buf139; del buf139  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_28(c_void_p(buf140.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del convolution_30
    del convolution_33
    del primals_252
    del primals_261
    del primals_262
    buf144 = buf142[1]
    del buf142
    buf145 = empty((512, ), device='cpu', dtype=torch.float32)
    buf146 = empty((512, ), device='cpu', dtype=torch.float32)
    buf147 = buf146; del buf146  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_29(c_void_p(buf147.data_ptr()), c_void_p(relu_29.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(buf145.data_ptr()))
    del buf143
    del convolution_32
    del primals_258
    del primals_259
    del relu_29
    buf151 = buf149[1]
    del buf149
    buf152 = empty((512, ), device='cpu', dtype=torch.float32)
    buf153 = empty((512, ), device='cpu', dtype=torch.float32)
    buf154 = buf153; del buf153  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_30(c_void_p(buf154.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(buf152.data_ptr()))
    del buf150
    del convolution_31
    del primals_255
    del primals_256
    del relu_28
    buf158 = buf156[1]
    del buf156
    buf161 = buf160; del buf160  # reuse
    buf162 = buf141; del buf141  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31(c_void_p(buf161.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf162.data_ptr()))
    del primals_253
    del primals_92
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf163 = aten.convolution_backward(buf162, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_91
    buf164 = buf163[0]
    buf165 = buf163[1]
    del buf163
    buf166 = empty((512, ), device='cpu', dtype=torch.float32)
    buf167 = empty((512, ), device='cpu', dtype=torch.float32)
    buf168 = buf167; del buf167  # reuse
    buf169 = buf164; del buf164  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf166.data_ptr()))
    del convolution_29
    del primals_249
    del primals_250
    del primals_89
    del relu_26
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf170 = aten.convolution_backward(buf169, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf169
    del primals_88
    buf171 = buf170[0]
    buf172 = buf170[1]
    del buf170
    buf173 = empty((512, ), device='cpu', dtype=torch.float32)
    buf174 = empty((512, ), device='cpu', dtype=torch.float32)
    buf175 = buf174; del buf174  # reuse
    buf176 = buf171; del buf171  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33(c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf173.data_ptr()))
    del convolution_28
    del primals_246
    del primals_247
    del primals_86
    del relu_25
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf177 = aten.convolution_backward(buf176, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf176
    del primals_85
    buf178 = buf177[0]
    buf179 = buf177[1]
    del buf177
    buf180 = buf137; del buf137  # reuse
    buf181 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf182 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf188 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf183 = buf182; del buf182  # reuse
    buf184 = buf162; del buf162  # reuse
    buf190 = buf94; del buf94  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_34(c_void_p(buf180.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf190.data_ptr()))
    del buf157
    del buf178
    del buf180
    del convolution_26
    del convolution_27
    del primals_240
    del primals_243
    del primals_244
    del primals_80
    del primals_83
    del relu_24
    del relu_27
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf185 = aten.convolution_backward(buf184, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf184
    del primals_82
    buf186 = buf185[0]
    buf187 = buf185[1]
    del buf185
    buf189 = buf188; del buf188  # reuse
    cpp_fused_native_batch_norm_backward_35(c_void_p(buf189.data_ptr()), c_void_p(primals_241.data_ptr()))
    del primals_241
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf191 = aten.convolution_backward(buf190, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf190
    del primals_79
    buf192 = buf191[0]
    buf193 = buf191[1]
    del buf191
    buf194 = empty((512, ), device='cpu', dtype=torch.float32)
    buf195 = empty((512, ), device='cpu', dtype=torch.float32)
    buf196 = buf195; del buf195  # reuse
    buf197 = buf192; del buf192  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf194.data_ptr()))
    del convolution_25
    del primals_237
    del primals_238
    del primals_77
    del relu_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf198 = aten.convolution_backward(buf197, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf197
    del primals_76
    buf199 = buf198[0]
    buf200 = buf198[1]
    del buf198
    buf201 = empty((512, ), device='cpu', dtype=torch.float32)
    buf202 = empty((512, ), device='cpu', dtype=torch.float32)
    buf203 = buf202; del buf202  # reuse
    buf204 = buf199; del buf199  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37(c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf201.data_ptr()))
    del convolution_24
    del primals_234
    del primals_235
    del primals_74
    del relu_22
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf205 = aten.convolution_backward(buf204, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_73
    buf206 = buf205[0]
    buf207 = buf205[1]
    del buf205
    buf208 = empty((512, ), device='cpu', dtype=torch.float32)
    buf209 = empty((512, ), device='cpu', dtype=torch.float32)
    buf210 = buf209; del buf209  # reuse
    buf211 = buf204; del buf204  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38(c_void_p(buf210.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf211.data_ptr()))
    del convolution_23
    del primals_231
    del primals_232
    del primals_71
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf212 = aten.convolution_backward(buf211, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_70
    buf213 = buf212[0]
    buf214 = buf212[1]
    del buf212
    buf215 = empty((256, ), device='cpu', dtype=torch.float32)
    buf216 = empty((256, ), device='cpu', dtype=torch.float32)
    buf217 = buf216; del buf216  # reuse
    buf218 = buf213; del buf213  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf215.data_ptr()))
    del convolution_22
    del primals_228
    del primals_229
    del primals_68
    del relu_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf219 = aten.convolution_backward(buf218, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf218
    del primals_67
    buf220 = buf219[0]
    buf221 = buf219[1]
    del buf219
    buf222 = empty((256, ), device='cpu', dtype=torch.float32)
    buf223 = empty((256, ), device='cpu', dtype=torch.float32)
    buf224 = buf223; del buf223  # reuse
    buf225 = buf220; del buf220  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40(c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf222.data_ptr()))
    del convolution_21
    del primals_225
    del primals_226
    del primals_65
    del relu_19
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf226 = aten.convolution_backward(buf225, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_64
    buf227 = buf226[0]
    buf228 = buf226[1]
    del buf226
    buf229 = buf186; del buf186  # reuse
    buf233 = buf211; del buf211  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_41(c_void_p(buf229.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf233.data_ptr()))
    del buf206
    del primals_62
    del relu_18
    del relu_21
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf234 = aten.convolution_backward(buf233, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_61
    buf235 = buf234[0]
    buf240 = buf225; del buf225  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42(c_void_p(relu_17.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf240.data_ptr()))
    del primals_59
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf241 = aten.convolution_backward(buf240, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del primals_58
    buf242 = buf241[0]
    buf247 = buf240; del buf240  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(relu_16.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf247.data_ptr()))
    del primals_56
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf248 = aten.convolution_backward(buf247, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf247
    del primals_55
    buf249 = buf248[0]
    buf230 = empty((512, ), device='cpu', dtype=torch.float32)
    buf231 = empty((512, ), device='cpu', dtype=torch.float32)
    buf251 = empty((512, ), device='cpu', dtype=torch.float32)
    buf252 = empty((512, ), device='cpu', dtype=torch.float32)
    buf232 = buf231; del buf231  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_44(c_void_p(buf232.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()))
    del convolution_17
    del convolution_20
    del primals_213
    del primals_222
    del primals_223
    buf236 = buf234[1]
    del buf234
    buf237 = empty((256, ), device='cpu', dtype=torch.float32)
    buf238 = empty((256, ), device='cpu', dtype=torch.float32)
    buf239 = buf238; del buf238  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_45(c_void_p(buf239.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(buf237.data_ptr()))
    del buf235
    del convolution_19
    del primals_219
    del primals_220
    del relu_17
    buf243 = buf241[1]
    del buf241
    buf244 = empty((256, ), device='cpu', dtype=torch.float32)
    buf245 = empty((256, ), device='cpu', dtype=torch.float32)
    buf246 = buf245; del buf245  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_46(c_void_p(buf246.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(buf244.data_ptr()))
    del buf242
    del convolution_18
    del primals_216
    del primals_217
    del relu_16
    buf250 = buf248[1]
    del buf248
    buf253 = buf252; del buf252  # reuse
    buf254 = buf233; del buf233  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_47(c_void_p(buf253.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf254.data_ptr()))
    del primals_214
    del primals_53
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf255 = aten.convolution_backward(buf254, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_52
    buf256 = buf255[0]
    buf257 = buf255[1]
    del buf255
    buf258 = empty((256, ), device='cpu', dtype=torch.float32)
    buf259 = empty((256, ), device='cpu', dtype=torch.float32)
    buf260 = buf259; del buf259  # reuse
    buf261 = buf256; del buf256  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48(c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf258.data_ptr()))
    del convolution_16
    del primals_210
    del primals_211
    del primals_50
    del relu_14
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf262 = aten.convolution_backward(buf261, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf261
    del primals_49
    buf263 = buf262[0]
    buf264 = buf262[1]
    del buf262
    buf265 = empty((256, ), device='cpu', dtype=torch.float32)
    buf266 = empty((256, ), device='cpu', dtype=torch.float32)
    buf267 = buf266; del buf266  # reuse
    buf268 = buf263; del buf263  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf265.data_ptr()))
    del convolution_15
    del primals_207
    del primals_208
    del primals_47
    del relu_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf269 = aten.convolution_backward(buf268, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf268
    del primals_46
    buf270 = buf269[0]
    buf271 = buf269[1]
    del buf269
    buf272 = buf229; del buf229  # reuse
    buf273 = empty((512, ), device='cpu', dtype=torch.float32)
    buf274 = empty((512, ), device='cpu', dtype=torch.float32)
    buf280 = empty((512, ), device='cpu', dtype=torch.float32)
    buf275 = buf274; del buf274  # reuse
    buf276 = buf254; del buf254  # reuse
    buf282 = buf227; del buf227  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_50(c_void_p(buf272.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf282.data_ptr()))
    del buf249
    del buf270
    del buf272
    del convolution_13
    del convolution_14
    del primals_201
    del primals_204
    del primals_205
    del primals_41
    del primals_44
    del relu_12
    del relu_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf277 = aten.convolution_backward(buf276, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf276
    del primals_43
    buf278 = buf277[0]
    buf279 = buf277[1]
    del buf277
    buf281 = buf280; del buf280  # reuse
    cpp_fused_native_batch_norm_backward_51(c_void_p(buf281.data_ptr()), c_void_p(primals_202.data_ptr()))
    del primals_202
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf283 = aten.convolution_backward(buf282, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf282
    del primals_40
    buf284 = buf283[0]
    buf285 = buf283[1]
    del buf283
    buf286 = empty((256, ), device='cpu', dtype=torch.float32)
    buf287 = empty((256, ), device='cpu', dtype=torch.float32)
    buf288 = buf287; del buf287  # reuse
    buf289 = buf284; del buf284  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52(c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf286.data_ptr()))
    del convolution_12
    del primals_198
    del primals_199
    del primals_38
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf290 = aten.convolution_backward(buf289, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf289
    del primals_37
    buf291 = buf290[0]
    buf292 = buf290[1]
    del buf290
    buf293 = empty((256, ), device='cpu', dtype=torch.float32)
    buf294 = empty((256, ), device='cpu', dtype=torch.float32)
    buf295 = buf294; del buf294  # reuse
    buf296 = buf291; del buf291  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53(c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf293.data_ptr()))
    del convolution_11
    del primals_195
    del primals_196
    del primals_35
    del relu_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf297 = aten.convolution_backward(buf296, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_34
    buf298 = buf297[0]
    buf299 = buf297[1]
    del buf297
    buf300 = empty((256, ), device='cpu', dtype=torch.float32)
    buf301 = empty((256, ), device='cpu', dtype=torch.float32)
    buf302 = buf301; del buf301  # reuse
    buf303 = buf296; del buf296  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_54(c_void_p(buf302.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf303.data_ptr()))
    del convolution_10
    del primals_192
    del primals_193
    del primals_32
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf304 = aten.convolution_backward(buf303, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_31
    buf305 = buf304[0]
    buf306 = buf304[1]
    del buf304
    buf307 = empty((128, ), device='cpu', dtype=torch.float32)
    buf308 = empty((128, ), device='cpu', dtype=torch.float32)
    buf309 = buf308; del buf308  # reuse
    buf310 = buf305; del buf305  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55(c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf307.data_ptr()))
    del convolution_9
    del primals_189
    del primals_190
    del primals_29
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf311 = aten.convolution_backward(buf310, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf310
    del primals_28
    buf312 = buf311[0]
    buf313 = buf311[1]
    del buf311
    buf314 = empty((128, ), device='cpu', dtype=torch.float32)
    buf315 = empty((128, ), device='cpu', dtype=torch.float32)
    buf316 = buf315; del buf315  # reuse
    buf317 = buf312; del buf312  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56(c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf314.data_ptr()))
    del convolution_8
    del primals_186
    del primals_187
    del primals_26
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf318 = aten.convolution_backward(buf317, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_25
    buf319 = buf318[0]
    buf320 = buf318[1]
    del buf318
    buf321 = buf278; del buf278  # reuse
    buf325 = buf303; del buf303  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_57(c_void_p(buf321.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf325.data_ptr()))
    del buf298
    del primals_23
    del relu_6
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf326 = aten.convolution_backward(buf325, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_22
    buf327 = buf326[0]
    buf332 = buf317; del buf317  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58(c_void_p(relu_5.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf332.data_ptr()))
    del primals_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf333 = aten.convolution_backward(buf332, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del primals_19
    buf334 = buf333[0]
    buf339 = buf332; del buf332  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59(c_void_p(relu_4.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf339.data_ptr()))
    del primals_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf340 = aten.convolution_backward(buf339, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf339
    del primals_16
    buf341 = buf340[0]
    buf322 = empty((256, ), device='cpu', dtype=torch.float32)
    buf323 = empty((256, ), device='cpu', dtype=torch.float32)
    buf343 = empty((256, ), device='cpu', dtype=torch.float32)
    buf344 = empty((256, ), device='cpu', dtype=torch.float32)
    buf350 = empty((256, ), device='cpu', dtype=torch.float32)
    buf324 = buf323; del buf323  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_60(c_void_p(buf324.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf350.data_ptr()))
    del convolution_3
    del convolution_4
    del convolution_7
    del primals_171
    del primals_174
    del primals_183
    del primals_184
    buf328 = buf326[1]
    del buf326
    buf329 = empty((128, ), device='cpu', dtype=torch.float32)
    buf330 = empty((128, ), device='cpu', dtype=torch.float32)
    buf331 = buf330; del buf330  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_61(c_void_p(buf331.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(buf329.data_ptr()))
    del buf327
    del convolution_6
    del primals_180
    del primals_181
    del relu_5
    buf335 = buf333[1]
    del buf333
    buf336 = empty((128, ), device='cpu', dtype=torch.float32)
    buf337 = empty((128, ), device='cpu', dtype=torch.float32)
    buf338 = buf337; del buf337  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_62(c_void_p(buf338.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(buf336.data_ptr()))
    del buf334
    del convolution_5
    del primals_177
    del primals_178
    del relu_4
    buf342 = buf340[1]
    del buf340
    buf345 = buf344; del buf344  # reuse
    buf346 = buf325; del buf325  # reuse
    buf352 = buf319; del buf319  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_63(c_void_p(buf345.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf352.data_ptr()))
    del buf321
    del buf341
    del primals_11
    del primals_14
    del primals_175
    del relu_3
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf347 = aten.convolution_backward(buf346, getitem, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf346
    del primals_13
    buf348 = buf347[0]
    buf349 = buf347[1]
    del buf347
    buf351 = buf350; del buf350  # reuse
    cpp_fused_native_batch_norm_backward_64(c_void_p(buf351.data_ptr()), c_void_p(primals_172.data_ptr()))
    del primals_172
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf353 = aten.convolution_backward(buf352, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf352
    del primals_10
    buf354 = buf353[0]
    buf355 = buf353[1]
    del buf353
    buf356 = empty((128, ), device='cpu', dtype=torch.float32)
    buf357 = empty((128, ), device='cpu', dtype=torch.float32)
    buf358 = buf357; del buf357  # reuse
    buf359 = buf354; del buf354  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_65(c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf356.data_ptr()))
    del convolution_2
    del primals_168
    del primals_169
    del primals_8
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf360 = aten.convolution_backward(buf359, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf359
    del primals_7
    buf361 = buf360[0]
    buf362 = buf360[1]
    del buf360
    buf363 = empty((128, ), device='cpu', dtype=torch.float32)
    buf364 = empty((128, ), device='cpu', dtype=torch.float32)
    buf365 = buf364; del buf364  # reuse
    buf366 = buf361; del buf361  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66(c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf363.data_ptr()))
    del convolution_1
    del primals_165
    del primals_166
    del primals_5
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf367 = aten.convolution_backward(buf366, getitem, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf366
    del getitem
    del primals_4
    buf368 = buf367[0]
    buf369 = buf367[1]
    del buf367
    buf370 = buf348; del buf348  # reuse
    cpp_fused_add_67(c_void_p(buf370.data_ptr()), c_void_p(buf368.data_ptr()))
    del buf368
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf371 = aten.max_pool2d_with_indices_backward(buf370, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1)
    del buf370
    del getitem_1
    buf372 = buf371
    del buf371
    buf373 = empty((64, ), device='cpu', dtype=torch.float32)
    buf374 = empty((64, ), device='cpu', dtype=torch.float32)
    buf375 = buf374; del buf374  # reuse
    buf376 = buf372; del buf372  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68(c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf373.data_ptr()))
    del convolution
    del primals_162
    del primals_163
    del primals_2
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf377 = aten.convolution_backward(buf376, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf376
    del primals_1
    del primals_321
    buf378 = buf377[1]
    return (buf378, buf375, buf373, buf369, buf365, buf363, buf362, buf358, buf356, buf355, buf351, buf343, buf349, buf345, buf343, buf342, buf338, buf336, buf335, buf331, buf329, buf328, buf324, buf322, buf320, buf316, buf314, buf313, buf309, buf307, buf306, buf302, buf300, buf299, buf295, buf293, buf292, buf288, buf286, buf285, buf281, buf273, buf279, buf275, buf273, buf271, buf267, buf265, buf264, buf260, buf258, buf257, buf253, buf251, buf250, buf246, buf244, buf243, buf239, buf237, buf236, buf232, buf230, buf228, buf224, buf222, buf221, buf217, buf215, buf214, buf210, buf208, buf207, buf203, buf201, buf200, buf196, buf194, buf193, buf189, buf181, buf187, buf183, buf181, buf179, buf175, buf173, buf172, buf168, buf166, buf165, buf161, buf159, buf158, buf154, buf152, buf151, buf147, buf145, buf144, buf140, buf138, buf136, buf132, buf130, buf129, buf125, buf123, buf122, buf118, buf116, buf115, buf111, buf109, buf108, buf104, buf102, buf101, buf97, buf95, buf93, buf89, buf87, buf86, buf82, buf80, buf79, buf75, buf73, buf72, buf68, buf66, buf65, buf61, buf59, buf58, buf54, buf46, buf52, buf48, buf46, buf44, buf40, buf38, buf37, buf33, buf31, buf30, buf26, buf24, buf23, buf19, buf17, buf16, buf12, buf10, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((128, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((128, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((256, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((256, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((256, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((256, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((512, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((512, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((512, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((512, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((512, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1024, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1024, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1024, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    getitem = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.int64)
    convolution_1 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    relu_36 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    relu_38 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    relu_39 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    relu_40 = rand_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    relu_41 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    relu_42 = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    relu_43 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    relu_44 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    relu_46 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    relu_47 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    view = rand_strided((4, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, convolution, relu, getitem, getitem_1, convolution_1, relu_1, convolution_2, relu_2, convolution_3, convolution_4, relu_3, convolution_5, relu_4, convolution_6, relu_5, convolution_7, relu_6, convolution_8, relu_7, convolution_9, relu_8, convolution_10, relu_9, convolution_11, relu_10, convolution_12, relu_11, convolution_13, convolution_14, relu_12, convolution_15, relu_13, convolution_16, relu_14, convolution_17, relu_15, convolution_18, relu_16, convolution_19, relu_17, convolution_20, relu_18, convolution_21, relu_19, convolution_22, relu_20, convolution_23, relu_21, convolution_24, relu_22, convolution_25, relu_23, convolution_26, convolution_27, relu_24, convolution_28, relu_25, convolution_29, relu_26, convolution_30, relu_27, convolution_31, relu_28, convolution_32, relu_29, convolution_33, relu_30, convolution_34, relu_31, convolution_35, relu_32, convolution_36, relu_33, convolution_37, relu_34, convolution_38, relu_35, convolution_39, relu_36, convolution_40, relu_37, convolution_41, relu_38, convolution_42, relu_39, convolution_43, relu_40, convolution_44, relu_41, convolution_45, convolution_46, relu_42, convolution_47, relu_43, convolution_48, relu_44, convolution_49, relu_45, convolution_50, relu_46, convolution_51, relu_47, convolution_52, view, permute_1, le, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resnext50_32x4d', benchmark_compiled_module)
