
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (1280L*x2) + (62720L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1280L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1280L*x2) + (62720L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (1280L*x1) + (62720L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1280L*x0)));
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
                        tmp15.store(out_ptr3 + static_cast<long>(x2 + (1280L*x1) + (62720L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (320L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (320L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (320L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (320L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
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
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.rsqrt();
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp2 * tmp9;
                tmp10.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = tmp8.rsqrt();
                auto tmp11 = tmp9 * tmp10;
                auto tmp12 = tmp4 * tmp11;
                tmp12.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_13 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr1 = in_out_ptr0;
    {
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
                float tmp_acc4 = 0;
                at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                float tmp_acc5 = 0;
                at::vec::Vectorized<float> tmp_acc5_vec = at::vec::Vectorized<float>(0);
                float tmp_acc6 = 0;
                at::vec::Vectorized<float> tmp_acc6_vec = at::vec::Vectorized<float>(0);
                float tmp_acc7 = 0;
                at::vec::Vectorized<float> tmp_acc7_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    auto tmp12 = tmp6 + tmp11;
                    auto tmp15 = tmp13 - tmp14;
                    auto tmp16 = tmp12 * tmp15;
                    auto tmp18 = tmp12 + tmp17;
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp22 = tmp18 * tmp21;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                    tmp_acc3_vec = tmp_acc3_vec + tmp10;
                    tmp_acc4_vec = tmp_acc4_vec + tmp12;
                    tmp_acc5_vec = tmp_acc5_vec + tmp16;
                    tmp_acc6_vec = tmp_acc6_vec + tmp18;
                    tmp_acc7_vec = tmp_acc7_vec + tmp22;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc4_vec.store(out_ptr4 + static_cast<long>(x0));
                tmp_acc5_vec.store(out_ptr5 + static_cast<long>(x0));
                tmp_acc6_vec.store(out_ptr6 + static_cast<long>(x0));
                tmp_acc7_vec.store(out_ptr7 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
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


cpp_fused_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp13 = tmp11 * tmp12;
                auto tmp14 = tmp6 * tmp13;
                tmp14.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (576L*x0)));
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
                tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (576L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (576L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (576L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (576L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (576L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_28 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (96L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (96L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                    tmp_acc3_vec = tmp_acc3_vec + tmp10;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (96L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.rsqrt();
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp2 * tmp9;
                tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (96L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.rsqrt();
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp2 * tmp9;
                tmp10.store(out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_40 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                float tmp_acc5 = 0;
                at::vec::Vectorized<float> tmp_acc5_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    auto tmp12 = tmp6 + tmp11;
                    auto tmp15 = tmp13 - tmp14;
                    auto tmp16 = tmp12 * tmp15;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                    tmp_acc3_vec = tmp_acc3_vec + tmp10;
                    tmp_acc4_vec = tmp_acc4_vec + tmp12;
                    tmp_acc5_vec = tmp_acc5_vec + tmp16;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc4_vec.store(out_ptr4 + static_cast<long>(x0));
                tmp_acc5_vec.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
''')


cpp_fused_native_batch_norm_backward_threshold_backward_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = tmp8.rsqrt();
                auto tmp11 = tmp9 * tmp10;
                auto tmp12 = tmp4 * tmp11;
                tmp12.store(in_out_ptr1 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_52 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_55 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc4 = 0;
                    at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc5 = 0;
                    at::vec::Vectorized<float> tmp_acc5_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        auto tmp12 = tmp6 + tmp11;
                        auto tmp15 = tmp13 - tmp14;
                        auto tmp16 = tmp12 * tmp15;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
                        tmp_acc3_vec = tmp_acc3_vec + tmp10;
                        tmp_acc4_vec = tmp_acc4_vec + tmp12;
                        tmp_acc5_vec = tmp_acc5_vec + tmp16;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc4_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc5_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = tmp8.rsqrt();
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    tmp12.store(in_out_ptr1 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (72L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (72L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_65 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_67 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
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
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_70 = async_compile.cpp('''
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
                       float* out_ptr4,
                       float* out_ptr5)
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc4 = 0;
                    at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc5 = 0;
                    at::vec::Vectorized<float> tmp_acc5_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        auto tmp12 = tmp6 + tmp11;
                        auto tmp15 = tmp13 - tmp14;
                        auto tmp16 = tmp12 * tmp15;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
                        tmp_acc3_vec = tmp_acc3_vec + tmp10;
                        tmp_acc4_vec = tmp_acc4_vec + tmp12;
                        tmp_acc5_vec = tmp_acc5_vec + tmp16;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc4_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc5_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = tmp8.rsqrt();
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    tmp12.store(in_out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (48L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (48L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (48L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (48L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, convolution, relu, convolution_1, relu_1, convolution_2, add_5, convolution_3, relu_2, convolution_4, relu_3, convolution_5, add_11, convolution_6, relu_4, convolution_7, relu_5, convolution_8, add_18, convolution_9, relu_6, convolution_10, relu_7, convolution_11, add_25, convolution_12, relu_8, convolution_13, relu_9, convolution_14, add_31, convolution_15, relu_10, convolution_16, relu_11, convolution_17, add_38, convolution_18, relu_12, convolution_19, relu_13, convolution_20, add_45, convolution_21, relu_14, convolution_22, relu_15, convolution_23, add_51, convolution_24, relu_16, convolution_25, relu_17, convolution_26, add_58, convolution_27, relu_18, convolution_28, relu_19, convolution_29, add_65, convolution_30, relu_20, convolution_31, relu_21, convolution_32, add_71, convolution_33, relu_22, convolution_34, relu_23, convolution_35, add_78, convolution_36, relu_24, convolution_37, relu_25, convolution_38, add_84, convolution_39, relu_26, convolution_40, relu_27, convolution_41, add_91, convolution_42, relu_28, convolution_43, relu_29, convolution_44, add_98, convolution_45, relu_30, convolution_46, relu_31, convolution_47, add_105, convolution_48, relu_32, convolution_49, relu_33, convolution_50, add_111, convolution_51, mean, permute_1, le, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_7, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_10, (48, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_11, (48, ), (1, ))
    assert_size_stride(primals_13, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (48, ), (1, ))
    assert_size_stride(primals_16, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_20, (72, ), (1, ))
    assert_size_stride(primals_22, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (72, ), (1, ))
    assert_size_stride(primals_25, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_28, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_29, (72, ), (1, ))
    assert_size_stride(primals_31, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (72, ), (1, ))
    assert_size_stride(primals_34, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_35, (24, ), (1, ))
    assert_size_stride(primals_37, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_38, (72, ), (1, ))
    assert_size_stride(primals_40, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_41, (72, ), (1, ))
    assert_size_stride(primals_43, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_44, (40, ), (1, ))
    assert_size_stride(primals_46, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_47, (120, ), (1, ))
    assert_size_stride(primals_49, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_50, (120, ), (1, ))
    assert_size_stride(primals_52, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_53, (40, ), (1, ))
    assert_size_stride(primals_55, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_56, (120, ), (1, ))
    assert_size_stride(primals_58, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_59, (120, ), (1, ))
    assert_size_stride(primals_61, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_62, (40, ), (1, ))
    assert_size_stride(primals_64, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_65, (240, ), (1, ))
    assert_size_stride(primals_67, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_68, (240, ), (1, ))
    assert_size_stride(primals_70, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_71, (80, ), (1, ))
    assert_size_stride(primals_73, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_74, (480, ), (1, ))
    assert_size_stride(primals_76, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_77, (480, ), (1, ))
    assert_size_stride(primals_79, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_80, (80, ), (1, ))
    assert_size_stride(primals_82, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_83, (480, ), (1, ))
    assert_size_stride(primals_85, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_86, (480, ), (1, ))
    assert_size_stride(primals_88, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_89, (80, ), (1, ))
    assert_size_stride(primals_91, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_92, (480, ), (1, ))
    assert_size_stride(primals_94, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_95, (480, ), (1, ))
    assert_size_stride(primals_97, (96, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_98, (96, ), (1, ))
    assert_size_stride(primals_100, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_101, (576, ), (1, ))
    assert_size_stride(primals_103, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_104, (576, ), (1, ))
    assert_size_stride(primals_106, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_107, (96, ), (1, ))
    assert_size_stride(primals_109, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_110, (576, ), (1, ))
    assert_size_stride(primals_112, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_113, (576, ), (1, ))
    assert_size_stride(primals_115, (192, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_118, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_119, (1152, ), (1, ))
    assert_size_stride(primals_121, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_122, (1152, ), (1, ))
    assert_size_stride(primals_124, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_125, (192, ), (1, ))
    assert_size_stride(primals_127, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_128, (1152, ), (1, ))
    assert_size_stride(primals_130, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_131, (1152, ), (1, ))
    assert_size_stride(primals_133, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_136, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_137, (1152, ), (1, ))
    assert_size_stride(primals_139, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_140, (1152, ), (1, ))
    assert_size_stride(primals_142, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_143, (192, ), (1, ))
    assert_size_stride(primals_145, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_146, (1152, ), (1, ))
    assert_size_stride(primals_148, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (1152, ), (1, ))
    assert_size_stride(primals_151, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_152, (320, ), (1, ))
    assert_size_stride(primals_154, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_155, (1280, ), (1, ))
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_162, (32, ), (1, ))
    assert_size_stride(primals_163, (32, ), (1, ))
    assert_size_stride(primals_165, (16, ), (1, ))
    assert_size_stride(primals_166, (16, ), (1, ))
    assert_size_stride(primals_168, (48, ), (1, ))
    assert_size_stride(primals_169, (48, ), (1, ))
    assert_size_stride(primals_171, (48, ), (1, ))
    assert_size_stride(primals_172, (48, ), (1, ))
    assert_size_stride(primals_174, (24, ), (1, ))
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_177, (72, ), (1, ))
    assert_size_stride(primals_178, (72, ), (1, ))
    assert_size_stride(primals_180, (72, ), (1, ))
    assert_size_stride(primals_181, (72, ), (1, ))
    assert_size_stride(primals_183, (24, ), (1, ))
    assert_size_stride(primals_184, (24, ), (1, ))
    assert_size_stride(primals_186, (72, ), (1, ))
    assert_size_stride(primals_187, (72, ), (1, ))
    assert_size_stride(primals_189, (72, ), (1, ))
    assert_size_stride(primals_190, (72, ), (1, ))
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_193, (24, ), (1, ))
    assert_size_stride(primals_195, (72, ), (1, ))
    assert_size_stride(primals_196, (72, ), (1, ))
    assert_size_stride(primals_198, (72, ), (1, ))
    assert_size_stride(primals_199, (72, ), (1, ))
    assert_size_stride(primals_201, (40, ), (1, ))
    assert_size_stride(primals_202, (40, ), (1, ))
    assert_size_stride(primals_204, (120, ), (1, ))
    assert_size_stride(primals_205, (120, ), (1, ))
    assert_size_stride(primals_207, (120, ), (1, ))
    assert_size_stride(primals_208, (120, ), (1, ))
    assert_size_stride(primals_210, (40, ), (1, ))
    assert_size_stride(primals_211, (40, ), (1, ))
    assert_size_stride(primals_213, (120, ), (1, ))
    assert_size_stride(primals_214, (120, ), (1, ))
    assert_size_stride(primals_216, (120, ), (1, ))
    assert_size_stride(primals_217, (120, ), (1, ))
    assert_size_stride(primals_219, (40, ), (1, ))
    assert_size_stride(primals_220, (40, ), (1, ))
    assert_size_stride(primals_222, (240, ), (1, ))
    assert_size_stride(primals_223, (240, ), (1, ))
    assert_size_stride(primals_225, (240, ), (1, ))
    assert_size_stride(primals_226, (240, ), (1, ))
    assert_size_stride(primals_228, (80, ), (1, ))
    assert_size_stride(primals_229, (80, ), (1, ))
    assert_size_stride(primals_231, (480, ), (1, ))
    assert_size_stride(primals_232, (480, ), (1, ))
    assert_size_stride(primals_234, (480, ), (1, ))
    assert_size_stride(primals_235, (480, ), (1, ))
    assert_size_stride(primals_237, (80, ), (1, ))
    assert_size_stride(primals_238, (80, ), (1, ))
    assert_size_stride(primals_240, (480, ), (1, ))
    assert_size_stride(primals_241, (480, ), (1, ))
    assert_size_stride(primals_243, (480, ), (1, ))
    assert_size_stride(primals_244, (480, ), (1, ))
    assert_size_stride(primals_246, (80, ), (1, ))
    assert_size_stride(primals_247, (80, ), (1, ))
    assert_size_stride(primals_249, (480, ), (1, ))
    assert_size_stride(primals_250, (480, ), (1, ))
    assert_size_stride(primals_252, (480, ), (1, ))
    assert_size_stride(primals_253, (480, ), (1, ))
    assert_size_stride(primals_255, (96, ), (1, ))
    assert_size_stride(primals_256, (96, ), (1, ))
    assert_size_stride(primals_258, (576, ), (1, ))
    assert_size_stride(primals_259, (576, ), (1, ))
    assert_size_stride(primals_261, (576, ), (1, ))
    assert_size_stride(primals_262, (576, ), (1, ))
    assert_size_stride(primals_264, (96, ), (1, ))
    assert_size_stride(primals_265, (96, ), (1, ))
    assert_size_stride(primals_267, (576, ), (1, ))
    assert_size_stride(primals_268, (576, ), (1, ))
    assert_size_stride(primals_270, (576, ), (1, ))
    assert_size_stride(primals_271, (576, ), (1, ))
    assert_size_stride(primals_273, (192, ), (1, ))
    assert_size_stride(primals_274, (192, ), (1, ))
    assert_size_stride(primals_276, (1152, ), (1, ))
    assert_size_stride(primals_277, (1152, ), (1, ))
    assert_size_stride(primals_279, (1152, ), (1, ))
    assert_size_stride(primals_280, (1152, ), (1, ))
    assert_size_stride(primals_282, (192, ), (1, ))
    assert_size_stride(primals_283, (192, ), (1, ))
    assert_size_stride(primals_285, (1152, ), (1, ))
    assert_size_stride(primals_286, (1152, ), (1, ))
    assert_size_stride(primals_288, (1152, ), (1, ))
    assert_size_stride(primals_289, (1152, ), (1, ))
    assert_size_stride(primals_291, (192, ), (1, ))
    assert_size_stride(primals_292, (192, ), (1, ))
    assert_size_stride(primals_294, (1152, ), (1, ))
    assert_size_stride(primals_295, (1152, ), (1, ))
    assert_size_stride(primals_297, (1152, ), (1, ))
    assert_size_stride(primals_298, (1152, ), (1, ))
    assert_size_stride(primals_300, (192, ), (1, ))
    assert_size_stride(primals_301, (192, ), (1, ))
    assert_size_stride(primals_303, (1152, ), (1, ))
    assert_size_stride(primals_304, (1152, ), (1, ))
    assert_size_stride(primals_306, (1152, ), (1, ))
    assert_size_stride(primals_307, (1152, ), (1, ))
    assert_size_stride(primals_309, (320, ), (1, ))
    assert_size_stride(primals_310, (320, ), (1, ))
    assert_size_stride(primals_312, (1280, ), (1, ))
    assert_size_stride(primals_313, (1280, ), (1, ))
    assert_size_stride(primals_315, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(relu, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(relu_1, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(add_5, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (4, 48, 112, 112), (602112, 1, 5376, 48))
    assert_size_stride(relu_2, (4, 48, 112, 112), (602112, 1, 5376, 48))
    assert_size_stride(convolution_4, (4, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(relu_3, (4, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_5, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_11, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_6, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_4, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_7, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_5, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_8, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_18, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_9, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_6, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_10, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_7, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_11, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_25, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_12, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_8, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_13, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(relu_9, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(convolution_14, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_31, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_15, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_10, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_16, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_11, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_17, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_38, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_18, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_12, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_19, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_13, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_20, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_45, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_21, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(relu_14, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_22, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(relu_15, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_23, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_51, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_24, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_16, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_25, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_17, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_26, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_58, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_27, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_18, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_28, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_19, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_29, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_65, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_30, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_20, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_31, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(relu_21, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_32, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(add_71, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_33, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(relu_22, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_34, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(relu_23, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_35, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(add_78, (4, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_36, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(relu_24, (4, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_37, (4, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(relu_25, (4, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(convolution_38, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_84, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_39, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_26, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_40, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_27, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_41, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_91, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_42, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_28, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_43, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_29, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_44, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_98, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_45, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_30, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_46, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_31, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_47, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_105, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_48, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_32, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_49, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(relu_33, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_50, (4, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(add_111, (4, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(convolution_51, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(mean, (4, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(le, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), mean, out=buf1)
    del mean
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf3 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf4 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf5 = buf4; del buf4  # reuse
    buf6 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_view_0(c_void_p(buf5.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del convolution_51
    del le
    del primals_155
    del primals_312
    del primals_313
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, add_111, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_111
    del buf6
    del primals_154
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((320, ), device='cpu', dtype=torch.float32)
    buf11 = empty((320, ), device='cpu', dtype=torch.float32)
    buf12 = buf11; del buf11  # reuse
    buf13 = buf8; del buf8  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_1(c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf10.data_ptr()))
    del convolution_50
    del primals_152
    del primals_309
    del primals_310
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf14 = aten.convolution_backward(buf13, relu_33, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_151
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf18 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf19 = buf18; del buf18  # reuse
    buf20 = buf15; del buf15  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf17.data_ptr()))
    del convolution_49
    del primals_149
    del primals_306
    del primals_307
    del relu_33
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf21 = aten.convolution_backward(buf20, relu_32, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf20
    del primals_148
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    buf24 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf25 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf26 = buf25; del buf25  # reuse
    buf27 = buf22; del buf22  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3(c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf24.data_ptr()))
    del convolution_48
    del primals_146
    del primals_303
    del primals_304
    del relu_32
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf28 = aten.convolution_backward(buf27, add_105, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_105
    del primals_145
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf34 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_4(c_void_p(buf29.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf34.data_ptr()))
    del primals_143
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf35 = aten.convolution_backward(buf34, relu_31, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_142
    buf36 = buf35[0]
    buf41 = buf27; del buf27  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(relu_31.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf41.data_ptr()))
    del primals_140
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf42 = aten.convolution_backward(buf41, relu_30, primals_139, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del primals_139
    buf43 = buf42[0]
    buf48 = buf41; del buf41  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(relu_30.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf48.data_ptr()))
    del primals_137
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf49 = aten.convolution_backward(buf48, add_98, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_98
    del primals_136
    buf50 = buf49[0]
    buf55 = buf34; del buf34  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_7(c_void_p(buf29.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf55.data_ptr()))
    del primals_134
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf56 = aten.convolution_backward(buf55, relu_29, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_133
    buf57 = buf56[0]
    buf62 = buf48; del buf48  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8(c_void_p(relu_29.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf62.data_ptr()))
    del primals_131
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf63 = aten.convolution_backward(buf62, relu_28, primals_130, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del primals_130
    buf64 = buf63[0]
    buf69 = buf62; del buf62  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(relu_28.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf69.data_ptr()))
    del primals_128
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf70 = aten.convolution_backward(buf69, add_91, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_91
    del primals_127
    buf71 = buf70[0]
    buf76 = buf55; del buf55  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_10(c_void_p(buf29.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf76.data_ptr()))
    del primals_125
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf77 = aten.convolution_backward(buf76, relu_27, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf76
    del primals_124
    buf78 = buf77[0]
    buf83 = buf69; del buf69  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(relu_27.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf83.data_ptr()))
    del primals_122
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf84 = aten.convolution_backward(buf83, relu_26, primals_121, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del primals_121
    buf85 = buf84[0]
    buf90 = buf83; del buf83  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(relu_26.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf90.data_ptr()))
    del primals_119
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf91 = aten.convolution_backward(buf90, add_84, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_84
    del buf90
    del primals_118
    buf92 = buf91[0]
    buf31 = empty((192, ), device='cpu', dtype=torch.float32)
    buf32 = empty((192, ), device='cpu', dtype=torch.float32)
    buf52 = empty((192, ), device='cpu', dtype=torch.float32)
    buf53 = empty((192, ), device='cpu', dtype=torch.float32)
    buf73 = empty((192, ), device='cpu', dtype=torch.float32)
    buf74 = empty((192, ), device='cpu', dtype=torch.float32)
    buf94 = empty((192, ), device='cpu', dtype=torch.float32)
    buf95 = empty((192, ), device='cpu', dtype=torch.float32)
    buf33 = buf32; del buf32  # reuse
    cpp_fused_add_native_batch_norm_backward_13(c_void_p(buf33.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    del convolution_38
    del convolution_41
    del convolution_44
    del convolution_47
    del primals_273
    del primals_282
    del primals_291
    del primals_300
    del primals_301
    buf37 = buf35[1]
    del buf35
    buf38 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf39 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf40 = buf39; del buf39  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_14(c_void_p(buf40.data_ptr()), c_void_p(relu_31.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(buf38.data_ptr()))
    del buf36
    del convolution_46
    del primals_297
    del primals_298
    del relu_31
    buf44 = buf42[1]
    del buf42
    buf45 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf46 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf47 = buf46; del buf46  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_15(c_void_p(buf47.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(buf45.data_ptr()))
    del buf43
    del convolution_45
    del primals_294
    del primals_295
    del relu_30
    buf51 = buf49[1]
    del buf49
    buf54 = buf53; del buf53  # reuse
    cpp_fused_native_batch_norm_backward_16(c_void_p(buf54.data_ptr()), c_void_p(primals_292.data_ptr()))
    del primals_292
    buf58 = buf56[1]
    del buf56
    buf59 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf60 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf61 = buf60; del buf60  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_17(c_void_p(buf61.data_ptr()), c_void_p(relu_29.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(buf59.data_ptr()))
    del buf57
    del convolution_43
    del primals_288
    del primals_289
    del relu_29
    buf65 = buf63[1]
    del buf63
    buf66 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf67 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf68 = buf67; del buf67  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_18(c_void_p(buf68.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(buf66.data_ptr()))
    del buf64
    del convolution_42
    del primals_285
    del primals_286
    del relu_28
    buf72 = buf70[1]
    del buf70
    buf75 = buf74; del buf74  # reuse
    cpp_fused_native_batch_norm_backward_19(c_void_p(buf75.data_ptr()), c_void_p(primals_283.data_ptr()))
    del primals_283
    buf79 = buf77[1]
    del buf77
    buf80 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf81 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf82 = buf81; del buf81  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_20(c_void_p(buf82.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(buf80.data_ptr()))
    del buf78
    del convolution_40
    del primals_279
    del primals_280
    del relu_27
    buf86 = buf84[1]
    del buf84
    buf87 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf88 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf89 = buf88; del buf88  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_21(c_void_p(buf89.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(buf87.data_ptr()))
    del buf85
    del convolution_39
    del primals_276
    del primals_277
    del relu_26
    buf93 = buf91[1]
    del buf91
    buf96 = buf95; del buf95  # reuse
    buf97 = buf29; del buf29  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_22(c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_116.data_ptr()))
    del buf50
    del buf71
    del buf92
    del primals_116
    del primals_274
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf98 = aten.convolution_backward(buf97, relu_25, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf97
    del primals_115
    buf99 = buf98[0]
    buf100 = buf98[1]
    del buf98
    buf101 = empty((576, ), device='cpu', dtype=torch.float32)
    buf102 = empty((576, ), device='cpu', dtype=torch.float32)
    buf103 = buf102; del buf102  # reuse
    buf104 = buf99; del buf99  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf101.data_ptr()))
    del convolution_37
    del primals_113
    del primals_270
    del primals_271
    del relu_25
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf105 = aten.convolution_backward(buf104, relu_24, primals_112, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 576, [True, True, False])
    del buf104
    del primals_112
    buf106 = buf105[0]
    buf107 = buf105[1]
    del buf105
    buf108 = empty((576, ), device='cpu', dtype=torch.float32)
    buf109 = empty((576, ), device='cpu', dtype=torch.float32)
    buf110 = buf109; del buf109  # reuse
    buf111 = buf106; del buf106  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf108.data_ptr()))
    del convolution_36
    del primals_110
    del primals_267
    del primals_268
    del relu_24
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf112 = aten.convolution_backward(buf111, add_78, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_78
    del primals_109
    buf113 = buf112[0]
    buf114 = buf112[1]
    del buf112
    buf118 = empty_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_25(c_void_p(buf113.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf118.data_ptr()))
    del primals_107
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf119 = aten.convolution_backward(buf118, relu_23, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf118
    del primals_106
    buf120 = buf119[0]
    buf125 = buf111; del buf111  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_26(c_void_p(relu_23.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf125.data_ptr()))
    del primals_104
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf126 = aten.convolution_backward(buf125, relu_22, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
    del primals_103
    buf127 = buf126[0]
    buf132 = buf125; del buf125  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(relu_22.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf132.data_ptr()))
    del primals_101
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf133 = aten.convolution_backward(buf132, add_71, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_71
    del buf132
    del primals_100
    buf134 = buf133[0]
    buf115 = empty((96, ), device='cpu', dtype=torch.float32)
    buf116 = empty((96, ), device='cpu', dtype=torch.float32)
    buf136 = empty((96, ), device='cpu', dtype=torch.float32)
    buf137 = empty((96, ), device='cpu', dtype=torch.float32)
    buf117 = buf116; del buf116  # reuse
    cpp_fused_add_native_batch_norm_backward_28(c_void_p(buf117.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()))
    del convolution_32
    del convolution_35
    del primals_255
    del primals_264
    del primals_265
    buf121 = buf119[1]
    del buf119
    buf122 = empty((576, ), device='cpu', dtype=torch.float32)
    buf123 = empty((576, ), device='cpu', dtype=torch.float32)
    buf124 = buf123; del buf123  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_29(c_void_p(buf124.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(buf122.data_ptr()))
    del buf120
    del convolution_34
    del primals_261
    del primals_262
    del relu_23
    buf128 = buf126[1]
    del buf126
    buf129 = empty((576, ), device='cpu', dtype=torch.float32)
    buf130 = empty((576, ), device='cpu', dtype=torch.float32)
    buf131 = buf130; del buf130  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_30(c_void_p(buf131.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(buf129.data_ptr()))
    del buf127
    del convolution_33
    del primals_258
    del primals_259
    del relu_22
    buf135 = buf133[1]
    del buf133
    buf138 = buf137; del buf137  # reuse
    buf139 = buf113; del buf113  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_31(c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(primals_98.data_ptr()))
    del buf134
    del primals_256
    del primals_98
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf140 = aten.convolution_backward(buf139, relu_21, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf139
    del primals_97
    buf141 = buf140[0]
    buf142 = buf140[1]
    del buf140
    buf143 = empty((480, ), device='cpu', dtype=torch.float32)
    buf144 = empty((480, ), device='cpu', dtype=torch.float32)
    buf145 = buf144; del buf144  # reuse
    buf146 = buf141; del buf141  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf143.data_ptr()))
    del convolution_31
    del primals_252
    del primals_253
    del primals_95
    del relu_21
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf147 = aten.convolution_backward(buf146, relu_20, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf146
    del primals_94
    buf148 = buf147[0]
    buf149 = buf147[1]
    del buf147
    buf150 = empty((480, ), device='cpu', dtype=torch.float32)
    buf151 = empty((480, ), device='cpu', dtype=torch.float32)
    buf152 = buf151; del buf151  # reuse
    buf153 = buf148; del buf148  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33(c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf150.data_ptr()))
    del convolution_30
    del primals_249
    del primals_250
    del primals_92
    del relu_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf154 = aten.convolution_backward(buf153, add_65, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_65
    del primals_91
    buf155 = buf154[0]
    buf156 = buf154[1]
    del buf154
    buf160 = reinterpret_tensor(buf13, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf13  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_34(c_void_p(buf155.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf160.data_ptr()))
    del primals_89
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf161 = aten.convolution_backward(buf160, relu_19, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_88
    buf162 = buf161[0]
    buf167 = buf153; del buf153  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35(c_void_p(relu_19.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf167.data_ptr()))
    del primals_86
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf168 = aten.convolution_backward(buf167, relu_18, primals_85, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False])
    del primals_85
    buf169 = buf168[0]
    buf174 = buf167; del buf167  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(relu_18.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf174.data_ptr()))
    del primals_83
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf175 = aten.convolution_backward(buf174, add_58, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_58
    del primals_82
    buf176 = buf175[0]
    buf181 = buf160; del buf160  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_37(c_void_p(buf155.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf181.data_ptr()))
    del primals_80
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf182 = aten.convolution_backward(buf181, relu_17, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf181
    del primals_79
    buf183 = buf182[0]
    buf188 = buf174; del buf174  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38(c_void_p(relu_17.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf188.data_ptr()))
    del primals_77
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf189 = aten.convolution_backward(buf188, relu_16, primals_76, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False])
    del primals_76
    buf190 = buf189[0]
    buf195 = buf188; del buf188  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(relu_16.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf195.data_ptr()))
    del primals_74
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf196 = aten.convolution_backward(buf195, add_51, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_51
    del buf195
    del primals_73
    buf197 = buf196[0]
    buf157 = empty((80, ), device='cpu', dtype=torch.float32)
    buf158 = empty((80, ), device='cpu', dtype=torch.float32)
    buf178 = empty((80, ), device='cpu', dtype=torch.float32)
    buf179 = empty((80, ), device='cpu', dtype=torch.float32)
    buf199 = empty((80, ), device='cpu', dtype=torch.float32)
    buf200 = empty((80, ), device='cpu', dtype=torch.float32)
    buf159 = buf158; del buf158  # reuse
    cpp_fused_add_native_batch_norm_backward_40(c_void_p(buf159.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    del convolution_23
    del convolution_26
    del convolution_29
    del primals_228
    del primals_237
    del primals_246
    del primals_247
    buf163 = buf161[1]
    del buf161
    buf164 = empty((480, ), device='cpu', dtype=torch.float32)
    buf165 = empty((480, ), device='cpu', dtype=torch.float32)
    buf166 = buf165; del buf165  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_41(c_void_p(buf166.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf164.data_ptr()))
    del buf162
    del convolution_28
    del primals_243
    del primals_244
    del relu_19
    buf170 = buf168[1]
    del buf168
    buf171 = empty((480, ), device='cpu', dtype=torch.float32)
    buf172 = empty((480, ), device='cpu', dtype=torch.float32)
    buf173 = buf172; del buf172  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_42(c_void_p(buf173.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(buf171.data_ptr()))
    del buf169
    del convolution_27
    del primals_240
    del primals_241
    del relu_18
    buf177 = buf175[1]
    del buf175
    buf180 = buf179; del buf179  # reuse
    cpp_fused_native_batch_norm_backward_43(c_void_p(buf180.data_ptr()), c_void_p(primals_238.data_ptr()))
    del primals_238
    buf184 = buf182[1]
    del buf182
    buf185 = empty((480, ), device='cpu', dtype=torch.float32)
    buf186 = empty((480, ), device='cpu', dtype=torch.float32)
    buf187 = buf186; del buf186  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_44(c_void_p(buf187.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(buf185.data_ptr()))
    del buf183
    del convolution_25
    del primals_234
    del primals_235
    del relu_17
    buf191 = buf189[1]
    del buf189
    buf192 = empty((480, ), device='cpu', dtype=torch.float32)
    buf193 = empty((480, ), device='cpu', dtype=torch.float32)
    buf194 = buf193; del buf193  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_45(c_void_p(buf194.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(buf192.data_ptr()))
    del convolution_24
    del primals_231
    del primals_232
    del relu_16
    buf198 = buf196[1]
    del buf196
    buf201 = buf200; del buf200  # reuse
    buf202 = buf155; del buf155  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_46(c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(primals_71.data_ptr()))
    del buf176
    del buf197
    del primals_229
    del primals_71
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf203 = aten.convolution_backward(buf202, relu_15, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf202
    del primals_70
    buf204 = buf203[0]
    buf205 = buf203[1]
    del buf203
    buf206 = empty((240, ), device='cpu', dtype=torch.float32)
    buf207 = empty((240, ), device='cpu', dtype=torch.float32)
    buf208 = buf207; del buf207  # reuse
    buf209 = buf204; del buf204  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47(c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf206.data_ptr()))
    del convolution_22
    del primals_225
    del primals_226
    del primals_68
    del relu_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf210 = aten.convolution_backward(buf209, relu_14, primals_67, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf209
    del primals_67
    buf211 = buf210[0]
    buf212 = buf210[1]
    del buf210
    buf213 = empty((240, ), device='cpu', dtype=torch.float32)
    buf214 = empty((240, ), device='cpu', dtype=torch.float32)
    buf215 = buf214; del buf214  # reuse
    buf216 = buf211; del buf211  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48(c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf213.data_ptr()))
    del convolution_21
    del primals_222
    del primals_223
    del primals_65
    del relu_14
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf217 = aten.convolution_backward(buf216, add_45, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_45
    del buf216
    del primals_64
    buf218 = buf217[0]
    buf219 = buf217[1]
    del buf217
    buf223 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_49(c_void_p(buf218.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf223.data_ptr()))
    del primals_62
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf224 = aten.convolution_backward(buf223, relu_13, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_61
    buf225 = buf224[0]
    buf230 = reinterpret_tensor(buf190, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf190  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50(c_void_p(relu_13.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf230.data_ptr()))
    del primals_59
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf231 = aten.convolution_backward(buf230, relu_12, primals_58, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del primals_58
    buf232 = buf231[0]
    buf237 = buf230; del buf230  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51(c_void_p(relu_12.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf237.data_ptr()))
    del primals_56
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf238 = aten.convolution_backward(buf237, add_38, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_38
    del primals_55
    buf239 = buf238[0]
    buf244 = buf223; del buf223  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_52(c_void_p(buf218.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf244.data_ptr()))
    del primals_53
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf245 = aten.convolution_backward(buf244, relu_11, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf244
    del primals_52
    buf246 = buf245[0]
    buf251 = buf237; del buf237  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53(c_void_p(relu_11.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf251.data_ptr()))
    del primals_50
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf252 = aten.convolution_backward(buf251, relu_10, primals_49, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del primals_49
    buf253 = buf252[0]
    buf258 = buf251; del buf251  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54(c_void_p(relu_10.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf258.data_ptr()))
    del primals_47
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf259 = aten.convolution_backward(buf258, add_31, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_31
    del buf258
    del primals_46
    buf260 = buf259[0]
    buf220 = empty((40, ), device='cpu', dtype=torch.float32)
    buf221 = empty((40, ), device='cpu', dtype=torch.float32)
    buf241 = empty((40, ), device='cpu', dtype=torch.float32)
    buf242 = empty((40, ), device='cpu', dtype=torch.float32)
    buf262 = empty((40, ), device='cpu', dtype=torch.float32)
    buf263 = empty((40, ), device='cpu', dtype=torch.float32)
    buf222 = buf221; del buf221  # reuse
    cpp_fused_add_native_batch_norm_backward_55(c_void_p(buf222.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()))
    del convolution_14
    del convolution_17
    del convolution_20
    del primals_201
    del primals_210
    del primals_219
    del primals_220
    buf226 = buf224[1]
    del buf224
    buf227 = empty((120, ), device='cpu', dtype=torch.float32)
    buf228 = empty((120, ), device='cpu', dtype=torch.float32)
    buf229 = buf228; del buf228  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_56(c_void_p(buf229.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(buf227.data_ptr()))
    del buf225
    del convolution_19
    del primals_216
    del primals_217
    del relu_13
    buf233 = buf231[1]
    del buf231
    buf234 = empty((120, ), device='cpu', dtype=torch.float32)
    buf235 = empty((120, ), device='cpu', dtype=torch.float32)
    buf236 = buf235; del buf235  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_57(c_void_p(buf236.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(buf234.data_ptr()))
    del buf232
    del convolution_18
    del primals_213
    del primals_214
    del relu_12
    buf240 = buf238[1]
    del buf238
    buf243 = buf242; del buf242  # reuse
    cpp_fused_native_batch_norm_backward_58(c_void_p(buf243.data_ptr()), c_void_p(primals_211.data_ptr()))
    del primals_211
    buf247 = buf245[1]
    del buf245
    buf248 = empty((120, ), device='cpu', dtype=torch.float32)
    buf249 = empty((120, ), device='cpu', dtype=torch.float32)
    buf250 = buf249; del buf249  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_59(c_void_p(buf250.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(buf248.data_ptr()))
    del buf246
    del convolution_16
    del primals_207
    del primals_208
    del relu_11
    buf254 = buf252[1]
    del buf252
    buf255 = empty((120, ), device='cpu', dtype=torch.float32)
    buf256 = empty((120, ), device='cpu', dtype=torch.float32)
    buf257 = buf256; del buf256  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_60(c_void_p(buf257.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(buf255.data_ptr()))
    del buf253
    del convolution_15
    del primals_204
    del primals_205
    del relu_10
    buf261 = buf259[1]
    del buf259
    buf264 = buf263; del buf263  # reuse
    buf265 = buf218; del buf218  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_61(c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(primals_44.data_ptr()))
    del buf239
    del buf260
    del primals_202
    del primals_44
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf266 = aten.convolution_backward(buf265, relu_9, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf265
    del primals_43
    buf267 = buf266[0]
    buf268 = buf266[1]
    del buf266
    buf269 = empty((72, ), device='cpu', dtype=torch.float32)
    buf270 = empty((72, ), device='cpu', dtype=torch.float32)
    buf271 = buf270; del buf270  # reuse
    buf272 = buf267; del buf267  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62(c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf269.data_ptr()))
    del convolution_13
    del primals_198
    del primals_199
    del primals_41
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf273 = aten.convolution_backward(buf272, relu_8, primals_40, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False])
    del buf272
    del primals_40
    buf274 = buf273[0]
    buf275 = buf273[1]
    del buf273
    buf276 = empty((72, ), device='cpu', dtype=torch.float32)
    buf277 = empty((72, ), device='cpu', dtype=torch.float32)
    buf278 = buf277; del buf277  # reuse
    buf279 = buf274; del buf274  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_63(c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf276.data_ptr()))
    del convolution_12
    del primals_195
    del primals_196
    del primals_38
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf280 = aten.convolution_backward(buf279, add_25, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_25
    del primals_37
    buf281 = buf280[0]
    buf282 = buf280[1]
    del buf280
    buf286 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_64(c_void_p(buf281.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf286.data_ptr()))
    del primals_35
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf287 = aten.convolution_backward(buf286, relu_7, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_34
    buf288 = buf287[0]
    buf293 = buf279; del buf279  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_65(c_void_p(relu_7.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf293.data_ptr()))
    del primals_32
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf294 = aten.convolution_backward(buf293, relu_6, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False])
    del primals_31
    buf295 = buf294[0]
    buf300 = buf293; del buf293  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66(c_void_p(relu_6.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf300.data_ptr()))
    del primals_29
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf301 = aten.convolution_backward(buf300, add_18, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_18
    del primals_28
    buf302 = buf301[0]
    buf307 = buf286; del buf286  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_67(c_void_p(buf281.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf307.data_ptr()))
    del primals_26
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf308 = aten.convolution_backward(buf307, relu_5, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf307
    del primals_25
    buf309 = buf308[0]
    buf314 = buf300; del buf300  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68(c_void_p(relu_5.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf314.data_ptr()))
    del primals_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf315 = aten.convolution_backward(buf314, relu_4, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False])
    del primals_22
    buf316 = buf315[0]
    buf321 = buf314; del buf314  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69(c_void_p(relu_4.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf321.data_ptr()))
    del primals_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf322 = aten.convolution_backward(buf321, add_11, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_11
    del buf321
    del primals_19
    buf323 = buf322[0]
    buf283 = empty((24, ), device='cpu', dtype=torch.float32)
    buf284 = empty((24, ), device='cpu', dtype=torch.float32)
    buf304 = empty((24, ), device='cpu', dtype=torch.float32)
    buf305 = empty((24, ), device='cpu', dtype=torch.float32)
    buf325 = empty((24, ), device='cpu', dtype=torch.float32)
    buf326 = empty((24, ), device='cpu', dtype=torch.float32)
    buf285 = buf284; del buf284  # reuse
    cpp_fused_add_native_batch_norm_backward_70(c_void_p(buf285.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del convolution_11
    del convolution_5
    del convolution_8
    del primals_174
    del primals_183
    del primals_192
    del primals_193
    buf289 = buf287[1]
    del buf287
    buf290 = empty((72, ), device='cpu', dtype=torch.float32)
    buf291 = empty((72, ), device='cpu', dtype=torch.float32)
    buf292 = buf291; del buf291  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_71(c_void_p(buf292.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(buf290.data_ptr()))
    del buf288
    del convolution_10
    del primals_189
    del primals_190
    del relu_7
    buf296 = buf294[1]
    del buf294
    buf297 = empty((72, ), device='cpu', dtype=torch.float32)
    buf298 = empty((72, ), device='cpu', dtype=torch.float32)
    buf299 = buf298; del buf298  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_72(c_void_p(buf299.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf297.data_ptr()))
    del buf295
    del convolution_9
    del primals_186
    del primals_187
    del relu_6
    buf303 = buf301[1]
    del buf301
    buf306 = buf305; del buf305  # reuse
    cpp_fused_native_batch_norm_backward_73(c_void_p(buf306.data_ptr()), c_void_p(primals_184.data_ptr()))
    del primals_184
    buf310 = buf308[1]
    del buf308
    buf311 = empty((72, ), device='cpu', dtype=torch.float32)
    buf312 = empty((72, ), device='cpu', dtype=torch.float32)
    buf313 = buf312; del buf312  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_74(c_void_p(buf313.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(buf311.data_ptr()))
    del buf309
    del convolution_7
    del primals_180
    del primals_181
    del relu_5
    buf317 = buf315[1]
    del buf315
    buf318 = empty((72, ), device='cpu', dtype=torch.float32)
    buf319 = empty((72, ), device='cpu', dtype=torch.float32)
    buf320 = buf319; del buf319  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_75(c_void_p(buf320.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(buf318.data_ptr()))
    del buf316
    del convolution_6
    del primals_177
    del primals_178
    del relu_4
    buf324 = buf322[1]
    del buf322
    buf327 = buf326; del buf326  # reuse
    buf328 = buf281; del buf281  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_76(c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(primals_17.data_ptr()))
    del buf302
    del buf323
    del primals_17
    del primals_175
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf329 = aten.convolution_backward(buf328, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf328
    del primals_16
    buf330 = buf329[0]
    buf331 = buf329[1]
    del buf329
    buf332 = empty((48, ), device='cpu', dtype=torch.float32)
    buf333 = empty((48, ), device='cpu', dtype=torch.float32)
    buf334 = buf333; del buf333  # reuse
    buf335 = buf330; del buf330  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77(c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf332.data_ptr()))
    del convolution_4
    del primals_14
    del primals_171
    del primals_172
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf336 = aten.convolution_backward(buf335, relu_2, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False])
    del buf335
    del primals_13
    buf337 = buf336[0]
    buf338 = buf336[1]
    del buf336
    buf339 = empty((48, ), device='cpu', dtype=torch.float32)
    buf340 = empty((48, ), device='cpu', dtype=torch.float32)
    buf341 = buf340; del buf340  # reuse
    buf342 = buf337; del buf337  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_78(c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf339.data_ptr()))
    del convolution_3
    del primals_11
    del primals_168
    del primals_169
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf343 = aten.convolution_backward(buf342, add_5, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_5
    del buf342
    del primals_10
    buf344 = buf343[0]
    buf345 = buf343[1]
    del buf343
    buf346 = empty((16, ), device='cpu', dtype=torch.float32)
    buf347 = empty((16, ), device='cpu', dtype=torch.float32)
    buf348 = buf347; del buf347  # reuse
    buf349 = buf344; del buf344  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_79(c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf346.data_ptr()))
    del convolution_2
    del primals_165
    del primals_166
    del primals_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf350 = aten.convolution_backward(buf349, relu_1, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf349
    del primals_7
    buf351 = buf350[0]
    buf352 = buf350[1]
    del buf350
    buf353 = empty((32, ), device='cpu', dtype=torch.float32)
    buf354 = empty((32, ), device='cpu', dtype=torch.float32)
    buf355 = buf354; del buf354  # reuse
    buf356 = buf351; del buf351  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_80(c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf353.data_ptr()))
    del convolution_1
    del primals_162
    del primals_163
    del primals_5
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf357 = aten.convolution_backward(buf356, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf356
    del primals_4
    buf358 = buf357[0]
    buf359 = buf357[1]
    del buf357
    buf360 = empty((32, ), device='cpu', dtype=torch.float32)
    buf361 = empty((32, ), device='cpu', dtype=torch.float32)
    buf362 = buf361; del buf361  # reuse
    buf363 = buf358; del buf358  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81(c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf360.data_ptr()))
    del convolution
    del primals_159
    del primals_160
    del primals_2
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf364 = aten.convolution_backward(buf363, primals_315, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf363
    del primals_1
    del primals_315
    buf365 = buf364[1]
    return (buf365, buf362, buf360, buf359, buf355, buf353, buf352, buf348, buf346, buf345, buf341, buf339, buf338, buf334, buf332, buf331, buf327, buf325, buf324, buf320, buf318, buf317, buf313, buf311, buf310, buf306, buf304, buf303, buf299, buf297, buf296, buf292, buf290, buf289, buf285, buf283, buf282, buf278, buf276, buf275, buf271, buf269, buf268, buf264, buf262, buf261, buf257, buf255, buf254, buf250, buf248, buf247, buf243, buf241, buf240, buf236, buf234, buf233, buf229, buf227, buf226, buf222, buf220, buf219, buf215, buf213, buf212, buf208, buf206, buf205, buf201, buf199, buf198, buf194, buf192, buf191, buf187, buf185, buf184, buf180, buf178, buf177, buf173, buf171, buf170, buf166, buf164, buf163, buf159, buf157, buf156, buf152, buf150, buf149, buf145, buf143, buf142, buf138, buf136, buf135, buf131, buf129, buf128, buf124, buf122, buf121, buf117, buf115, buf114, buf110, buf108, buf107, buf103, buf101, buf100, buf96, buf94, buf93, buf89, buf87, buf86, buf82, buf80, buf79, buf75, buf73, buf72, buf68, buf66, buf65, buf61, buf59, buf58, buf54, buf52, buf51, buf47, buf45, buf44, buf40, buf38, buf37, buf33, buf31, buf30, buf26, buf24, buf23, buf19, buf17, buf16, buf12, buf10, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((48, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((96, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((192, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    add_5 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((4, 48, 112, 112), (602112, 1, 5376, 48), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((4, 48, 112, 112), (602112, 1, 5376, 48), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    add_11 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    add_18 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    add_25 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    add_31 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    add_38 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    add_45 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_51 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_58 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_65 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    add_71 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    add_78 = rand_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    add_84 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    add_91 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    add_98 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    add_105 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cpu', dtype=torch.float32)
    add_111 = rand_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    mean = rand_strided((4, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, convolution, relu, convolution_1, relu_1, convolution_2, add_5, convolution_3, relu_2, convolution_4, relu_3, convolution_5, add_11, convolution_6, relu_4, convolution_7, relu_5, convolution_8, add_18, convolution_9, relu_6, convolution_10, relu_7, convolution_11, add_25, convolution_12, relu_8, convolution_13, relu_9, convolution_14, add_31, convolution_15, relu_10, convolution_16, relu_11, convolution_17, add_38, convolution_18, relu_12, convolution_19, relu_13, convolution_20, add_45, convolution_21, relu_14, convolution_22, relu_15, convolution_23, add_51, convolution_24, relu_16, convolution_25, relu_17, convolution_26, add_58, convolution_27, relu_18, convolution_28, relu_19, convolution_29, add_65, convolution_30, relu_20, convolution_31, relu_21, convolution_32, add_71, convolution_33, relu_22, convolution_34, relu_23, convolution_35, add_78, convolution_36, relu_24, convolution_37, relu_25, convolution_38, add_84, convolution_39, relu_26, convolution_40, relu_27, convolution_41, add_91, convolution_42, relu_28, convolution_43, relu_29, convolution_44, add_98, convolution_45, relu_30, convolution_46, relu_31, convolution_47, add_105, convolution_48, relu_32, convolution_49, relu_33, convolution_50, add_111, convolution_51, mean, permute_1, le, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mnasnet1_0', benchmark_compiled_module)
