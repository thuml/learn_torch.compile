
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
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
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
                        tmp15.store(out_ptr3 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
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
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1920L + x0 + (2144L*x1)));
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
            auto tmp2 = static_cast<float>(1e-05);
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1920L + x1 + (2144L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = tmp8.rsqrt();
                auto tmp11 = tmp9 * tmp10;
                auto tmp12 = tmp4 * tmp11;
                tmp12.store(out_ptr2 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_2 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1696L + x0 + (2144L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1696L + x1 + (2144L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
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
                tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1472L + x0 + (2144L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1472L + x1 + (2144L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
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
                tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1248L + x0 + (2144L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1248L + x1 + (2144L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
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
                tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x0 + (2144L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x1 + (2144L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
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
                tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
            }
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
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2144L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2144L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1664L + x0 + (1888L*x1)));
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
            auto tmp2 = static_cast<float>(1e-05);
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1664L + x1 + (1888L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = tmp8.rsqrt();
                auto tmp11 = tmp9 * tmp10;
                auto tmp12 = tmp4 * tmp11;
                tmp12.store(out_ptr2 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1440L + x0 + (1888L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1440L + x1 + (1888L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
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
                tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1216L + x0 + (1888L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1216L + x1 + (1888L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
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
                tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(992L + x0 + (1888L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(992L + x1 + (1888L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
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
                tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x0 + (1888L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x1 + (1888L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
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
                tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1888L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1536L + x0 + (1728L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1536L + x1 + (1728L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = tmp8.rsqrt();
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    tmp12.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1344L + x0 + (1728L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1344L + x1 + (1728L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1152L + x0 + (1728L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1152L + x1 + (1728L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(960L + x0 + (1728L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(960L + x1 + (1728L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x0 + (1728L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x1 + (1728L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1728L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1728L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1280L + x0 + (1472L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1280L + x1 + (1472L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = tmp8.rsqrt();
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    tmp12.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_21 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1088L + x0 + (1472L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1088L + x1 + (1472L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(896L + x0 + (1472L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(896L + x1 + (1472L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(704L + x0 + (1472L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(704L + x1 + (1472L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (1472L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1472L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1472L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.cpp('''
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(896L + x0 + (1056L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(896L + x1 + (1056L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = tmp8.rsqrt();
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    tmp12.store(out_ptr2 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(736L + x0 + (1056L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(736L + x1 + (1056L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(576L + x0 + (1056L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(576L + x1 + (1056L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(416L + x0 + (1056L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(416L + x1 + (1056L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (1056L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (1056L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1056L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(640L + x0 + (768L*x1)));
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
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(640L + x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = tmp8.rsqrt();
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    tmp12.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (768L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(384L + x0 + (768L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(384L + x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (768L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (768L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr2 = in_out_ptr1;
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.cpp('''
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, convolution_3, relu_3, convolution_4, relu_4, convolution_5, relu_5, convolution_6, relu_6, convolution_7, cat, convolution_8, relu_8, getitem, getitem_1, convolution_9, relu_9, convolution_10, relu_10, convolution_11, relu_11, convolution_12, relu_12, convolution_13, cat_1, convolution_14, relu_14, getitem_2, getitem_3, convolution_15, relu_15, convolution_16, relu_16, convolution_17, relu_17, convolution_18, relu_18, convolution_19, cat_2, convolution_20, relu_20, convolution_21, relu_21, convolution_22, relu_22, convolution_23, relu_23, convolution_24, relu_24, convolution_25, cat_3, convolution_26, relu_26, getitem_4, getitem_5, convolution_27, relu_27, convolution_28, relu_28, convolution_29, relu_29, convolution_30, relu_30, convolution_31, cat_4, convolution_32, relu_32, convolution_33, relu_33, convolution_34, relu_34, convolution_35, relu_35, convolution_36, relu_36, convolution_37, cat_5, convolution_38, clone, permute_1, le, le_1, le_7, le_13, le_19, le_25, le_31, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_25, (160, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (224, ), (1, ))
    assert_size_stride(primals_57, (224, ), (1, ))
    assert_size_stride(primals_59, (224, ), (1, ))
    assert_size_stride(primals_61, (224, ), (1, ))
    assert_size_stride(primals_63, (224, ), (1, ))
    assert_size_stride(primals_65, (1024, ), (1, ))
    assert_size_stride(primals_67, (224, ), (1, ))
    assert_size_stride(primals_69, (224, ), (1, ))
    assert_size_stride(primals_71, (224, ), (1, ))
    assert_size_stride(primals_73, (224, ), (1, ))
    assert_size_stride(primals_75, (224, ), (1, ))
    assert_size_stride(primals_77, (1024, ), (1, ))
    assert_size_stride(primals_79, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_80, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_81, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_82, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_83, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_84, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_85, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_86, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_87, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_88, (160, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_89, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_90, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_91, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_92, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_93, (512, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_94, (192, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_95, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_96, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_97, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_98, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_99, (768, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(primals_100, (192, 768, 3, 3), (6912, 1, 2304, 768))
    assert_size_stride(primals_101, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_102, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_103, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_104, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_105, (768, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(primals_106, (224, 768, 3, 3), (6912, 1, 2304, 768))
    assert_size_stride(primals_107, (224, 224, 3, 3), (2016, 1, 672, 224))
    assert_size_stride(primals_108, (224, 224, 3, 3), (2016, 1, 672, 224))
    assert_size_stride(primals_109, (224, 224, 3, 3), (2016, 1, 672, 224))
    assert_size_stride(primals_110, (224, 224, 3, 3), (2016, 1, 672, 224))
    assert_size_stride(primals_111, (1024, 1888, 1, 1), (1888, 1, 1, 1))
    assert_size_stride(primals_112, (224, 1024, 3, 3), (9216, 1, 3072, 1024))
    assert_size_stride(primals_113, (224, 224, 3, 3), (2016, 1, 672, 224))
    assert_size_stride(primals_114, (224, 224, 3, 3), (2016, 1, 672, 224))
    assert_size_stride(primals_115, (224, 224, 3, 3), (2016, 1, 672, 224))
    assert_size_stride(primals_116, (224, 224, 3, 3), (2016, 1, 672, 224))
    assert_size_stride(primals_117, (1024, 2144, 1, 1), (2144, 1, 1, 1))
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (64, ), (1, ))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (160, ), (1, ))
    assert_size_stride(primals_139, (160, ), (1, ))
    assert_size_stride(primals_140, (160, ), (1, ))
    assert_size_stride(primals_141, (160, ), (1, ))
    assert_size_stride(primals_142, (160, ), (1, ))
    assert_size_stride(primals_143, (160, ), (1, ))
    assert_size_stride(primals_144, (160, ), (1, ))
    assert_size_stride(primals_145, (160, ), (1, ))
    assert_size_stride(primals_146, (160, ), (1, ))
    assert_size_stride(primals_147, (160, ), (1, ))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (192, ), (1, ))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_153, (192, ), (1, ))
    assert_size_stride(primals_154, (192, ), (1, ))
    assert_size_stride(primals_155, (192, ), (1, ))
    assert_size_stride(primals_156, (192, ), (1, ))
    assert_size_stride(primals_157, (192, ), (1, ))
    assert_size_stride(primals_158, (192, ), (1, ))
    assert_size_stride(primals_159, (192, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, ), (1, ))
    assert_size_stride(primals_162, (192, ), (1, ))
    assert_size_stride(primals_163, (192, ), (1, ))
    assert_size_stride(primals_164, (192, ), (1, ))
    assert_size_stride(primals_165, (192, ), (1, ))
    assert_size_stride(primals_166, (192, ), (1, ))
    assert_size_stride(primals_167, (192, ), (1, ))
    assert_size_stride(primals_168, (192, ), (1, ))
    assert_size_stride(primals_169, (192, ), (1, ))
    assert_size_stride(primals_170, (192, ), (1, ))
    assert_size_stride(primals_171, (192, ), (1, ))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (224, ), (1, ))
    assert_size_stride(primals_175, (224, ), (1, ))
    assert_size_stride(primals_176, (224, ), (1, ))
    assert_size_stride(primals_177, (224, ), (1, ))
    assert_size_stride(primals_178, (224, ), (1, ))
    assert_size_stride(primals_179, (224, ), (1, ))
    assert_size_stride(primals_180, (224, ), (1, ))
    assert_size_stride(primals_181, (224, ), (1, ))
    assert_size_stride(primals_182, (224, ), (1, ))
    assert_size_stride(primals_183, (224, ), (1, ))
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (224, ), (1, ))
    assert_size_stride(primals_187, (224, ), (1, ))
    assert_size_stride(primals_188, (224, ), (1, ))
    assert_size_stride(primals_189, (224, ), (1, ))
    assert_size_stride(primals_190, (224, ), (1, ))
    assert_size_stride(primals_191, (224, ), (1, ))
    assert_size_stride(primals_192, (224, ), (1, ))
    assert_size_stride(primals_193, (224, ), (1, ))
    assert_size_stride(primals_194, (224, ), (1, ))
    assert_size_stride(primals_195, (224, ), (1, ))
    assert_size_stride(primals_196, (1024, ), (1, ))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_198, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(relu, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_1, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(relu_1, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_2, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_2, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_3, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_3, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_4, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_4, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_5, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_5, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_6, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(relu_6, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_7, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(cat, (4, 768, 56, 56), (2408448, 1, 43008, 768))
    assert_size_stride(convolution_8, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(relu_8, (4, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(getitem, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_1, (4, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_9, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(relu_9, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_10, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(relu_10, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_11, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(relu_11, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_12, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(relu_12, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(convolution_13, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(cat_1, (4, 1056, 28, 28), (827904, 1, 29568, 1056))
    assert_size_stride(convolution_14, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(relu_14, (4, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(getitem_2, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_3, (4, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_15, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(relu_15, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_16, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(relu_16, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_17, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(relu_17, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_18, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(relu_18, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_19, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(cat_2, (4, 1472, 14, 14), (288512, 1, 20608, 1472))
    assert_size_stride(convolution_20, (4, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(relu_20, (4, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(convolution_21, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(relu_21, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_22, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(relu_22, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_23, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(relu_23, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_24, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(relu_24, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_25, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(cat_3, (4, 1728, 14, 14), (338688, 1, 24192, 1728))
    assert_size_stride(convolution_26, (4, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(relu_26, (4, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(getitem_4, (4, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(getitem_5, (4, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_27, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(relu_27, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_28, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(relu_28, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_29, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(relu_29, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_30, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(relu_30, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_31, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(cat_4, (4, 1888, 7, 7), (92512, 1, 13216, 1888))
    assert_size_stride(convolution_32, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(relu_32, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_33, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(relu_33, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_34, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(relu_34, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_35, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(relu_35, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_36, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(relu_36, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(convolution_37, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(cat_5, (4, 2144, 7, 7), (105056, 1, 15008, 2144))
    assert_size_stride(convolution_38, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(clone, (4, 1024), (1024, 1))
    assert_size_stride(permute_1, (1000, 1024), (1024, 1))
    assert_size_stride(le, (4, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(le_1, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(le_7, (4, 224, 7, 7), (10976, 1, 1568, 224))
    assert_size_stride(le_13, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(le_19, (4, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(le_25, (4, 160, 28, 28), (125440, 1, 4480, 160))
    assert_size_stride(le_31, (4, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), clone, out=buf1)
    del clone
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf3 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf4 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf5 = buf4; del buf4  # reuse
    buf6 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_view_0(c_void_p(buf5.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del convolution_38
    del le
    del primals_196
    del primals_197
    del primals_77
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, cat_5, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf6
    del cat_5
    del primals_117
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((224, ), device='cpu', dtype=torch.float32)
    buf11 = empty((224, ), device='cpu', dtype=torch.float32)
    buf12 = buf11; del buf11  # reuse
    buf13 = empty_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1(c_void_p(buf12.data_ptr()), c_void_p(le_1.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf13.data_ptr()))
    del convolution_37
    del le_1
    del primals_194
    del primals_195
    del primals_75
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf14 = aten.convolution_backward(buf13, relu_36, primals_116, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf13
    del primals_116
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty((224, ), device='cpu', dtype=torch.float32)
    buf18 = empty((224, ), device='cpu', dtype=torch.float32)
    buf19 = buf18; del buf18  # reuse
    buf20 = buf15; del buf15  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_2(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf17.data_ptr()))
    del convolution_36
    del primals_192
    del primals_193
    del primals_73
    del relu_36
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf21 = aten.convolution_backward(buf20, relu_35, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf20
    del primals_115
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    buf24 = empty((224, ), device='cpu', dtype=torch.float32)
    buf25 = empty((224, ), device='cpu', dtype=torch.float32)
    buf26 = buf25; del buf25  # reuse
    buf27 = buf22; del buf22  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_3(c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf24.data_ptr()))
    del convolution_35
    del primals_190
    del primals_191
    del primals_71
    del relu_35
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf28 = aten.convolution_backward(buf27, relu_34, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf27
    del primals_114
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf31 = empty((224, ), device='cpu', dtype=torch.float32)
    buf32 = empty((224, ), device='cpu', dtype=torch.float32)
    buf33 = buf32; del buf32  # reuse
    buf34 = buf29; del buf29  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf31.data_ptr()))
    del convolution_34
    del primals_188
    del primals_189
    del primals_69
    del relu_34
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf35 = aten.convolution_backward(buf34, relu_33, primals_113, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf34
    del primals_113
    buf36 = buf35[0]
    buf37 = buf35[1]
    del buf35
    buf38 = empty((224, ), device='cpu', dtype=torch.float32)
    buf39 = empty((224, ), device='cpu', dtype=torch.float32)
    buf40 = buf39; del buf39  # reuse
    buf41 = buf36; del buf36  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf38.data_ptr()))
    del convolution_33
    del primals_186
    del primals_187
    del primals_67
    del relu_33
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf42 = aten.convolution_backward(buf41, relu_32, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_112
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    buf45 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf46 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf47 = buf46; del buf46  # reuse
    buf48 = buf43; del buf43  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf45.data_ptr()))
    del buf8
    del convolution_32
    del primals_184
    del primals_185
    del primals_65
    del relu_32
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf49 = aten.convolution_backward(buf48, cat_4, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf48
    del cat_4
    del primals_111
    buf50 = buf49[0]
    buf51 = buf49[1]
    del buf49
    buf52 = empty((224, ), device='cpu', dtype=torch.float32)
    buf53 = empty((224, ), device='cpu', dtype=torch.float32)
    buf54 = buf53; del buf53  # reuse
    buf55 = buf41; del buf41  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7(c_void_p(buf54.data_ptr()), c_void_p(le_7.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf55.data_ptr()))
    del convolution_31
    del le_7
    del primals_182
    del primals_183
    del primals_63
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf56 = aten.convolution_backward(buf55, relu_30, primals_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf55
    del primals_110
    buf57 = buf56[0]
    buf58 = buf56[1]
    del buf56
    buf59 = empty((224, ), device='cpu', dtype=torch.float32)
    buf60 = empty((224, ), device='cpu', dtype=torch.float32)
    buf61 = buf60; del buf60  # reuse
    buf62 = buf57; del buf57  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8(c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf59.data_ptr()))
    del convolution_30
    del primals_180
    del primals_181
    del primals_61
    del relu_30
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf63 = aten.convolution_backward(buf62, relu_29, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf62
    del primals_109
    buf64 = buf63[0]
    buf65 = buf63[1]
    del buf63
    buf66 = empty((224, ), device='cpu', dtype=torch.float32)
    buf67 = empty((224, ), device='cpu', dtype=torch.float32)
    buf68 = buf67; del buf67  # reuse
    buf69 = buf64; del buf64  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(relu_29.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf66.data_ptr()))
    del convolution_29
    del primals_178
    del primals_179
    del primals_59
    del relu_29
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf70 = aten.convolution_backward(buf69, relu_28, primals_108, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf69
    del primals_108
    buf71 = buf70[0]
    buf72 = buf70[1]
    del buf70
    buf73 = empty((224, ), device='cpu', dtype=torch.float32)
    buf74 = empty((224, ), device='cpu', dtype=torch.float32)
    buf75 = buf74; del buf74  # reuse
    buf76 = buf71; del buf71  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf73.data_ptr()))
    del convolution_28
    del primals_176
    del primals_177
    del primals_57
    del relu_28
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf77 = aten.convolution_backward(buf76, relu_27, primals_107, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf76
    del primals_107
    buf78 = buf77[0]
    buf79 = buf77[1]
    del buf77
    buf80 = empty((224, ), device='cpu', dtype=torch.float32)
    buf81 = empty((224, ), device='cpu', dtype=torch.float32)
    buf82 = buf81; del buf81  # reuse
    buf83 = buf78; del buf78  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf80.data_ptr()))
    del convolution_27
    del primals_174
    del primals_175
    del primals_55
    del relu_27
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf84 = aten.convolution_backward(buf83, getitem_4, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf83
    del getitem_4
    del primals_106
    buf85 = buf84[0]
    buf86 = buf84[1]
    del buf84
    buf87 = buf85; del buf85  # reuse
    cpp_fused_add_12(c_void_p(buf87.data_ptr()), c_void_p(buf50.data_ptr()))
    del buf50
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf88 = aten.max_pool2d_with_indices_backward(buf87, relu_26, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_5)
    del getitem_5
    buf89 = buf88
    del buf88
    buf90 = empty((768, ), device='cpu', dtype=torch.float32)
    buf91 = empty((768, ), device='cpu', dtype=torch.float32)
    buf92 = buf91; del buf91  # reuse
    buf93 = buf89; del buf89  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf90.data_ptr()))
    del convolution_26
    del primals_172
    del primals_173
    del primals_53
    del relu_26
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf94 = aten.convolution_backward(buf93, cat_3, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf93
    del cat_3
    del primals_105
    buf95 = buf94[0]
    buf96 = buf94[1]
    del buf94
    buf97 = empty((192, ), device='cpu', dtype=torch.float32)
    buf98 = empty((192, ), device='cpu', dtype=torch.float32)
    buf99 = buf98; del buf98  # reuse
    buf100 = reinterpret_tensor(buf87, (4, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf87  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf99.data_ptr()), c_void_p(le_13.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf100.data_ptr()))
    del convolution_25
    del le_13
    del primals_170
    del primals_171
    del primals_51
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf101 = aten.convolution_backward(buf100, relu_24, primals_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf100
    del primals_104
    buf102 = buf101[0]
    buf103 = buf101[1]
    del buf101
    buf104 = empty((192, ), device='cpu', dtype=torch.float32)
    buf105 = empty((192, ), device='cpu', dtype=torch.float32)
    buf106 = buf105; del buf105  # reuse
    buf107 = buf102; del buf102  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf104.data_ptr()))
    del convolution_24
    del primals_168
    del primals_169
    del primals_49
    del relu_24
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf108 = aten.convolution_backward(buf107, relu_23, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf107
    del primals_103
    buf109 = buf108[0]
    buf110 = buf108[1]
    del buf108
    buf111 = empty((192, ), device='cpu', dtype=torch.float32)
    buf112 = empty((192, ), device='cpu', dtype=torch.float32)
    buf113 = buf112; del buf112  # reuse
    buf114 = buf109; del buf109  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_16(c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf111.data_ptr()))
    del convolution_23
    del primals_166
    del primals_167
    del primals_47
    del relu_23
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf115 = aten.convolution_backward(buf114, relu_22, primals_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf114
    del primals_102
    buf116 = buf115[0]
    buf117 = buf115[1]
    del buf115
    buf118 = empty((192, ), device='cpu', dtype=torch.float32)
    buf119 = empty((192, ), device='cpu', dtype=torch.float32)
    buf120 = buf119; del buf119  # reuse
    buf121 = buf116; del buf116  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf118.data_ptr()))
    del convolution_22
    del primals_164
    del primals_165
    del primals_45
    del relu_22
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf122 = aten.convolution_backward(buf121, relu_21, primals_101, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf121
    del primals_101
    buf123 = buf122[0]
    buf124 = buf122[1]
    del buf122
    buf125 = empty((192, ), device='cpu', dtype=torch.float32)
    buf126 = empty((192, ), device='cpu', dtype=torch.float32)
    buf127 = buf126; del buf126  # reuse
    buf128 = buf123; del buf123  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf125.data_ptr()))
    del convolution_21
    del primals_162
    del primals_163
    del primals_43
    del relu_21
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf129 = aten.convolution_backward(buf128, relu_20, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_100
    buf130 = buf129[0]
    buf131 = buf129[1]
    del buf129
    buf132 = empty((768, ), device='cpu', dtype=torch.float32)
    buf133 = empty((768, ), device='cpu', dtype=torch.float32)
    buf134 = buf133; del buf133  # reuse
    buf135 = buf130; del buf130  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf132.data_ptr()))
    del buf95
    del convolution_20
    del primals_160
    del primals_161
    del primals_41
    del relu_20
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf136 = aten.convolution_backward(buf135, cat_2, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf135
    del cat_2
    del primals_99
    buf137 = buf136[0]
    buf138 = buf136[1]
    del buf136
    buf139 = empty((192, ), device='cpu', dtype=torch.float32)
    buf140 = empty((192, ), device='cpu', dtype=torch.float32)
    buf141 = buf140; del buf140  # reuse
    buf142 = buf128; del buf128  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(buf141.data_ptr()), c_void_p(le_19.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf142.data_ptr()))
    del convolution_19
    del le_19
    del primals_158
    del primals_159
    del primals_39
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf143 = aten.convolution_backward(buf142, relu_18, primals_98, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf142
    del primals_98
    buf144 = buf143[0]
    buf145 = buf143[1]
    del buf143
    buf146 = empty((192, ), device='cpu', dtype=torch.float32)
    buf147 = empty((192, ), device='cpu', dtype=torch.float32)
    buf148 = buf147; del buf147  # reuse
    buf149 = buf144; del buf144  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_21(c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf146.data_ptr()))
    del convolution_18
    del primals_156
    del primals_157
    del primals_37
    del relu_18
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf150 = aten.convolution_backward(buf149, relu_17, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf149
    del primals_97
    buf151 = buf150[0]
    buf152 = buf150[1]
    del buf150
    buf153 = empty((192, ), device='cpu', dtype=torch.float32)
    buf154 = empty((192, ), device='cpu', dtype=torch.float32)
    buf155 = buf154; del buf154  # reuse
    buf156 = buf151; del buf151  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf153.data_ptr()))
    del convolution_17
    del primals_154
    del primals_155
    del primals_35
    del relu_17
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf157 = aten.convolution_backward(buf156, relu_16, primals_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf156
    del primals_96
    buf158 = buf157[0]
    buf159 = buf157[1]
    del buf157
    buf160 = empty((192, ), device='cpu', dtype=torch.float32)
    buf161 = empty((192, ), device='cpu', dtype=torch.float32)
    buf162 = buf161; del buf161  # reuse
    buf163 = buf158; del buf158  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf160.data_ptr()))
    del convolution_16
    del primals_152
    del primals_153
    del primals_33
    del relu_16
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf164 = aten.convolution_backward(buf163, relu_15, primals_95, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf163
    del primals_95
    buf165 = buf164[0]
    buf166 = buf164[1]
    del buf164
    buf167 = empty((192, ), device='cpu', dtype=torch.float32)
    buf168 = empty((192, ), device='cpu', dtype=torch.float32)
    buf169 = buf168; del buf168  # reuse
    buf170 = buf165; del buf165  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf167.data_ptr()))
    del convolution_15
    del primals_150
    del primals_151
    del primals_31
    del relu_15
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf171 = aten.convolution_backward(buf170, getitem_2, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf170
    del getitem_2
    del primals_94
    buf172 = buf171[0]
    buf173 = buf171[1]
    del buf171
    buf174 = buf172; del buf172  # reuse
    cpp_fused_add_25(c_void_p(buf174.data_ptr()), c_void_p(buf137.data_ptr()))
    del buf137
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf175 = aten.max_pool2d_with_indices_backward(buf174, relu_14, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_3)
    del buf174
    del getitem_3
    buf176 = buf175
    del buf175
    buf177 = empty((512, ), device='cpu', dtype=torch.float32)
    buf178 = empty((512, ), device='cpu', dtype=torch.float32)
    buf179 = buf178; del buf178  # reuse
    buf180 = buf176; del buf176  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26(c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf177.data_ptr()))
    del convolution_14
    del primals_148
    del primals_149
    del primals_29
    del relu_14
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf181 = aten.convolution_backward(buf180, cat_1, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_1
    del primals_93
    buf182 = buf181[0]
    buf183 = buf181[1]
    del buf181
    buf184 = empty((160, ), device='cpu', dtype=torch.float32)
    buf185 = empty((160, ), device='cpu', dtype=torch.float32)
    buf186 = buf185; del buf185  # reuse
    buf187 = empty_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(buf186.data_ptr()), c_void_p(le_25.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf187.data_ptr()))
    del convolution_13
    del le_25
    del primals_146
    del primals_147
    del primals_27
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf188 = aten.convolution_backward(buf187, relu_12, primals_92, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf187
    del primals_92
    buf189 = buf188[0]
    buf190 = buf188[1]
    del buf188
    buf191 = empty((160, ), device='cpu', dtype=torch.float32)
    buf192 = empty((160, ), device='cpu', dtype=torch.float32)
    buf193 = buf192; del buf192  # reuse
    buf194 = buf189; del buf189  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_28(c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf191.data_ptr()))
    del convolution_12
    del primals_144
    del primals_145
    del primals_25
    del relu_12
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf195 = aten.convolution_backward(buf194, relu_11, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf194
    del primals_91
    buf196 = buf195[0]
    buf197 = buf195[1]
    del buf195
    buf198 = empty((160, ), device='cpu', dtype=torch.float32)
    buf199 = empty((160, ), device='cpu', dtype=torch.float32)
    buf200 = buf199; del buf199  # reuse
    buf201 = buf196; del buf196  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_29(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf198.data_ptr()))
    del convolution_11
    del primals_142
    del primals_143
    del primals_23
    del relu_11
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf202 = aten.convolution_backward(buf201, relu_10, primals_90, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf201
    del primals_90
    buf203 = buf202[0]
    buf204 = buf202[1]
    del buf202
    buf205 = empty((160, ), device='cpu', dtype=torch.float32)
    buf206 = empty((160, ), device='cpu', dtype=torch.float32)
    buf207 = buf206; del buf206  # reuse
    buf208 = buf203; del buf203  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_30(c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf205.data_ptr()))
    del convolution_10
    del primals_140
    del primals_141
    del primals_21
    del relu_10
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf209 = aten.convolution_backward(buf208, relu_9, primals_89, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf208
    del primals_89
    buf210 = buf209[0]
    buf211 = buf209[1]
    del buf209
    buf212 = empty((160, ), device='cpu', dtype=torch.float32)
    buf213 = empty((160, ), device='cpu', dtype=torch.float32)
    buf214 = buf213; del buf213  # reuse
    buf215 = buf210; del buf210  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31(c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf212.data_ptr()))
    del convolution_9
    del primals_138
    del primals_139
    del primals_19
    del relu_9
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf216 = aten.convolution_backward(buf215, getitem, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf215
    del getitem
    del primals_88
    buf217 = buf216[0]
    buf218 = buf216[1]
    del buf216
    buf219 = buf217; del buf217  # reuse
    cpp_fused_add_32(c_void_p(buf219.data_ptr()), c_void_p(buf182.data_ptr()))
    del buf182
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf220 = aten.max_pool2d_with_indices_backward(buf219, relu_8, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_1)
    del buf219
    del getitem_1
    buf221 = buf220
    del buf220
    buf222 = empty((256, ), device='cpu', dtype=torch.float32)
    buf223 = empty((256, ), device='cpu', dtype=torch.float32)
    buf224 = buf223; del buf223  # reuse
    buf225 = buf221; del buf221  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33(c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf222.data_ptr()))
    del convolution_8
    del primals_136
    del primals_137
    del primals_17
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf226 = aten.convolution_backward(buf225, cat, primals_87, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf225
    del cat
    del primals_87
    buf227 = buf226[0]
    buf228 = buf226[1]
    del buf226
    buf229 = empty((128, ), device='cpu', dtype=torch.float32)
    buf230 = empty((128, ), device='cpu', dtype=torch.float32)
    buf231 = buf230; del buf230  # reuse
    buf232 = reinterpret_tensor(buf180, (4, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf180  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34(c_void_p(buf231.data_ptr()), c_void_p(le_31.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf232.data_ptr()))
    del convolution_7
    del le_31
    del primals_134
    del primals_135
    del primals_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf233 = aten.convolution_backward(buf232, relu_6, primals_86, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf232
    del primals_86
    buf234 = buf233[0]
    buf235 = buf233[1]
    del buf233
    buf236 = empty((128, ), device='cpu', dtype=torch.float32)
    buf237 = empty((128, ), device='cpu', dtype=torch.float32)
    buf238 = buf237; del buf237  # reuse
    buf239 = buf234; del buf234  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_35(c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf236.data_ptr()))
    del convolution_6
    del primals_13
    del primals_132
    del primals_133
    del relu_6
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf240 = aten.convolution_backward(buf239, relu_5, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf239
    del primals_85
    buf241 = buf240[0]
    buf242 = buf240[1]
    del buf240
    buf243 = empty((128, ), device='cpu', dtype=torch.float32)
    buf244 = empty((128, ), device='cpu', dtype=torch.float32)
    buf245 = buf244; del buf244  # reuse
    buf246 = buf241; del buf241  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf243.data_ptr()))
    del convolution_5
    del primals_11
    del primals_130
    del primals_131
    del relu_5
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf247 = aten.convolution_backward(buf246, relu_4, primals_84, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf246
    del primals_84
    buf248 = buf247[0]
    buf249 = buf247[1]
    del buf247
    buf250 = empty((128, ), device='cpu', dtype=torch.float32)
    buf251 = empty((128, ), device='cpu', dtype=torch.float32)
    buf252 = buf251; del buf251  # reuse
    buf253 = buf248; del buf248  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_37(c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf250.data_ptr()))
    del convolution_4
    del primals_128
    del primals_129
    del primals_9
    del relu_4
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf254 = aten.convolution_backward(buf253, relu_3, primals_83, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf253
    del primals_83
    buf255 = buf254[0]
    buf256 = buf254[1]
    del buf254
    buf257 = empty((128, ), device='cpu', dtype=torch.float32)
    buf258 = empty((128, ), device='cpu', dtype=torch.float32)
    buf259 = buf258; del buf258  # reuse
    buf260 = buf255; del buf255  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38(c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf257.data_ptr()))
    del convolution_3
    del primals_126
    del primals_127
    del primals_7
    del relu_3
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf261 = aten.convolution_backward(buf260, relu_2, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf260
    del primals_82
    buf262 = buf261[0]
    buf263 = buf261[1]
    del buf261
    buf264 = empty((128, ), device='cpu', dtype=torch.float32)
    buf265 = empty((128, ), device='cpu', dtype=torch.float32)
    buf266 = buf265; del buf265  # reuse
    buf267 = buf262; del buf262  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf264.data_ptr()))
    del buf227
    del convolution_2
    del primals_124
    del primals_125
    del primals_5
    del relu_2
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf268 = aten.convolution_backward(buf267, relu_1, primals_81, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf267
    del primals_81
    buf269 = buf268[0]
    buf270 = buf268[1]
    del buf268
    buf271 = empty((64, ), device='cpu', dtype=torch.float32)
    buf272 = empty((64, ), device='cpu', dtype=torch.float32)
    buf273 = buf272; del buf272  # reuse
    buf274 = buf269; del buf269  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40(c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf271.data_ptr()))
    del convolution_1
    del primals_122
    del primals_123
    del primals_3
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf275 = aten.convolution_backward(buf274, relu, primals_80, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf274
    del primals_80
    buf276 = buf275[0]
    buf277 = buf275[1]
    del buf275
    buf278 = empty((64, ), device='cpu', dtype=torch.float32)
    buf279 = empty((64, ), device='cpu', dtype=torch.float32)
    buf280 = buf279; del buf279  # reuse
    buf281 = buf276; del buf276  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41(c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf278.data_ptr()))
    del convolution
    del primals_1
    del primals_120
    del primals_121
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf282 = aten.convolution_backward(buf281, primals_198, primals_79, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf281
    del primals_198
    del primals_79
    buf283 = buf282[1]
    return (buf280, buf278, buf273, buf271, buf266, buf264, buf259, buf257, buf252, buf250, buf245, buf243, buf238, buf236, buf231, buf229, buf224, buf222, buf214, buf212, buf207, buf205, buf200, buf198, buf193, buf191, buf186, buf184, buf179, buf177, buf169, buf167, buf162, buf160, buf155, buf153, buf148, buf146, buf141, buf139, buf134, buf132, buf127, buf125, buf120, buf118, buf113, buf111, buf106, buf104, buf99, buf97, buf92, buf90, buf82, buf80, buf75, buf73, buf68, buf66, buf61, buf59, buf54, buf52, buf47, buf45, buf40, buf38, buf33, buf31, buf26, buf24, buf19, buf17, buf12, buf10, buf5, buf3, buf283, buf277, buf270, buf263, buf256, buf249, buf242, buf235, buf228, buf218, buf211, buf204, buf197, buf190, buf183, buf173, buf166, buf159, buf152, buf145, buf138, buf131, buf124, buf117, buf110, buf103, buf96, buf86, buf79, buf72, buf65, buf58, buf51, buf44, buf37, buf30, buf23, buf16, buf9, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((160, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, 1056, 1, 1), (1056, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((192, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, 1472, 1, 1), (1472, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((192, 768, 3, 3), (6912, 1, 2304, 768), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, 1728, 1, 1), (1728, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((224, 768, 3, 3), (6912, 1, 2304, 768), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1024, 1888, 1, 1), (1888, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((224, 1024, 3, 3), (9216, 1, 3072, 1024), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((224, 224, 3, 3), (2016, 1, 672, 224), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1024, 2144, 1, 1), (2144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cat = rand_strided((4, 768, 56, 56), (2408448, 1, 43008, 768), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    getitem = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.int64)
    convolution_9 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((4, 1056, 28, 28), (827904, 1, 29568, 1056), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.int64)
    convolution_15 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((4, 1472, 14, 14), (288512, 1, 20608, 1472), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((4, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((4, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((4, 1728, 14, 14), (338688, 1, 24192, 1728), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((4, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((4, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    getitem_4 = rand_strided((4, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((4, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.int64)
    convolution_27 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((4, 1888, 7, 7), (92512, 1, 13216, 1888), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    relu_36 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.float32)
    cat_5 = rand_strided((4, 2144, 7, 7), (105056, 1, 15008, 2144), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    clone = rand_strided((4, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.bool)
    le_1 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.bool)
    le_7 = rand_strided((4, 224, 7, 7), (10976, 1, 1568, 224), device='cpu', dtype=torch.bool)
    le_13 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.bool)
    le_19 = rand_strided((4, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.bool)
    le_25 = rand_strided((4, 160, 28, 28), (125440, 1, 4480, 160), device='cpu', dtype=torch.bool)
    le_31 = rand_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, convolution_3, relu_3, convolution_4, relu_4, convolution_5, relu_5, convolution_6, relu_6, convolution_7, cat, convolution_8, relu_8, getitem, getitem_1, convolution_9, relu_9, convolution_10, relu_10, convolution_11, relu_11, convolution_12, relu_12, convolution_13, cat_1, convolution_14, relu_14, getitem_2, getitem_3, convolution_15, relu_15, convolution_16, relu_16, convolution_17, relu_17, convolution_18, relu_18, convolution_19, cat_2, convolution_20, relu_20, convolution_21, relu_21, convolution_22, relu_22, convolution_23, relu_23, convolution_24, relu_24, convolution_25, cat_3, convolution_26, relu_26, getitem_4, getitem_5, convolution_27, relu_27, convolution_28, relu_28, convolution_29, relu_29, convolution_30, relu_30, convolution_31, cat_4, convolution_32, relu_32, convolution_33, relu_33, convolution_34, relu_34, convolution_35, relu_35, convolution_36, relu_36, convolution_37, cat_5, convolution_38, clone, permute_1, le, le_1, le_7, le_13, le_19, le_25, le_31, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_vovnet', benchmark_compiled_module)
