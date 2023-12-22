
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
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
                        tmp25.store(out_ptr4 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (1024L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1024L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                tmp22.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_2 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (1024L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (1024L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                auto tmp11 = static_cast<float>(0.002551020408163265);
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
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                auto tmp11 = static_cast<float>(0.002551020408163265);
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
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(256);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (256L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(512);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-256L) + x1 + (256L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(768);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-512L) + x1 + (256L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(1024);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp33;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp4 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x2) + (100352L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp4 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
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
                        tmp27.store(out_ptr2 + static_cast<long>(x2 + (2048L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x0 + (1024L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x1 + (1024L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                tmp22.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (1024L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (1024L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                auto tmp11 = static_cast<float>(0.002551020408163265);
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
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                auto tmp11 = static_cast<float>(0.002551020408163265);
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
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(256);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (256L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(512);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-256L) + x1 + (256L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(768);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-512L) + x1 + (256L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(1024);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp33;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (2048L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(768L + x1 + (1024L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (1024L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 7L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 7L)) + (50176L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(768L + x1 + (1024L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (1024L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 7L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 7L)) + (50176L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(768L + x1 + (1024L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (1024L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 7L)) + (7168L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 7L)) + (50176L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(768L + x1 + (1024L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (1024L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 7L)) + (7168L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 7L)) + (50176L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp3 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer((1L + x2), 2L))));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer((1L + x3), 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp8 ? tmp1 : tmp9;
                            auto tmp12 = tmp11 / 9;
                            auto tmp13 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))));
                            auto tmp14 = tmp13 < tmp6;
                            auto tmp15 = tmp4 & tmp14;
                            auto tmp16 = decltype(tmp10)(tmp10 + tmp12);
                            auto tmp17 = tmp15 ? tmp16 : tmp10;
                            auto tmp19 = tmp18 / 9;
                            auto tmp20 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))));
                            auto tmp21 = tmp20 < tmp3;
                            auto tmp22 = tmp21 & tmp7;
                            auto tmp23 = decltype(tmp17)(tmp17 + tmp19);
                            auto tmp24 = tmp22 ? tmp23 : tmp17;
                            auto tmp26 = tmp25 / 9;
                            auto tmp27 = tmp21 & tmp14;
                            auto tmp28 = decltype(tmp24)(tmp24 + tmp26);
                            auto tmp29 = tmp27 ? tmp28 : tmp24;
                            out_ptr0[static_cast<long>(x3 + (14L*x2) + (196L*x1) + (50176L*x0))] = tmp29;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x0 + (1024L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(0.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                            auto tmp7 = tmp5 - tmp6;
                            auto tmp8 = tmp4 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                            tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                        tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
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
                        tmp22.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (1024L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (1024L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                tmp22.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                tmp22.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(256);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (256L*x1) + (50176L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(512);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = tmp9 & tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr2[static_cast<long>((-256L) + x2 + (256L*x1) + (50176L*x0))];
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp16 = tmp1 >= tmp10;
                        auto tmp17 = static_cast<long>(768);
                        auto tmp18 = tmp1 < tmp17;
                        auto tmp19 = tmp16 & tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr3[static_cast<long>((-512L) + x2 + (256L*x1) + (50176L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp1 >= tmp17;
                        auto tmp24 = static_cast<long>(1024);
                        auto tmp25 = tmp1 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr4[static_cast<long>((-150528L) + x1 + (196L*x2) + (50176L*x0))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = tmp19 ? tmp22 : tmp28;
                        auto tmp30 = tmp12 ? tmp15 : tmp29;
                        auto tmp31 = tmp5 ? tmp8 : tmp30;
                        auto tmp32 = static_cast<float>(0.0);
                        auto tmp33 = tmp0 ? tmp32 : tmp31;
                        out_ptr0[static_cast<long>(x2 + (1024L*x1) + (200704L*x0))] = tmp33;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_19 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(128);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(256);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-128L) + x1 + (128L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(384);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-256L) + x1 + (128L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(512);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp33;
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_23 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(128);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(256);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-128L) + x1 + (128L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(384);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-256L) + x1 + (128L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(512);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp33;
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_26 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_28 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(128);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(256);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-128L) + x1 + (128L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(384);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-256L) + x1 + (128L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(512);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp33;
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_33 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_34 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(128);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(256);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-128L) + x1 + (128L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(384);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-256L) + x1 + (128L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(512);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp33;
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x1 + (512L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_39 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(128);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(256);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-128L) + x1 + (128L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(384);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-256L) + x1 + (128L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(512);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp33;
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_42 = async_compile.cpp('''
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


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(384L + x1 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 14L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 14L)) + (100352L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(384L + x1 + (512L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (512L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 14L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 14L)) + (100352L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(384L + x1 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 14L)) + (7168L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 14L)) + (100352L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(384L + x1 + (512L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (512L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 14L)) + (7168L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 14L)) + (100352L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp3 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer((1L + x2), 2L))));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer((1L + x3), 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp8 ? tmp1 : tmp9;
                            auto tmp12 = tmp11 / 9;
                            auto tmp13 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))));
                            auto tmp14 = tmp13 < tmp6;
                            auto tmp15 = tmp4 & tmp14;
                            auto tmp16 = decltype(tmp10)(tmp10 + tmp12);
                            auto tmp17 = tmp15 ? tmp16 : tmp10;
                            auto tmp19 = tmp18 / 9;
                            auto tmp20 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))));
                            auto tmp21 = tmp20 < tmp3;
                            auto tmp22 = tmp21 & tmp7;
                            auto tmp23 = decltype(tmp17)(tmp17 + tmp19);
                            auto tmp24 = tmp22 ? tmp23 : tmp17;
                            auto tmp26 = tmp25 / 9;
                            auto tmp27 = tmp21 & tmp14;
                            auto tmp28 = decltype(tmp24)(tmp24 + tmp26);
                            auto tmp29 = tmp27 ? tmp28 : tmp24;
                            out_ptr0[static_cast<long>(x3 + (28L*x2) + (784L*x1) + (100352L*x0))] = tmp29;
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x0 + (512L*x1)));
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
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
                    tmp22.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_44 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (512L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_46 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (401408L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(128);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (128L*x1) + (100352L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(256);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = tmp9 & tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr2[static_cast<long>((-128L) + x2 + (128L*x1) + (100352L*x0))];
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp16 = tmp1 >= tmp10;
                        auto tmp17 = static_cast<long>(384);
                        auto tmp18 = tmp1 < tmp17;
                        auto tmp19 = tmp16 & tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr3[static_cast<long>((-256L) + x2 + (128L*x1) + (100352L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp1 >= tmp17;
                        auto tmp24 = static_cast<long>(512);
                        auto tmp25 = tmp1 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr4[static_cast<long>((-301056L) + x1 + (784L*x2) + (100352L*x0))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = tmp19 ? tmp22 : tmp28;
                        auto tmp30 = tmp12 ? tmp15 : tmp29;
                        auto tmp31 = tmp5 ? tmp8 : tmp30;
                        auto tmp32 = static_cast<float>(0.0);
                        auto tmp33 = tmp0 ? tmp32 : tmp31;
                        out_ptr0[static_cast<long>(x2 + (512L*x1) + (401408L*x0))] = tmp33;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_47 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_49 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_50 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(64);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (64L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(128);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-64L) + x1 + (64L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(192);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-128L) + x1 + (64L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(256);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp33;
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_54 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_55 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(64);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (64L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(128);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-64L) + x1 + (64L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(192);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-128L) + x1 + (64L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(256);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp33;
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_60 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
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


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_61 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(64);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (64L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(128);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-64L) + x1 + (64L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(192);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-128L) + x1 + (64L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(256);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp33;
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_63 = async_compile.cpp('''
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


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(192L + x1 + (256L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (256L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 28L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 28L)) + (200704L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(192L + x1 + (256L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (256L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 28L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 28L)) + (200704L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(192L + x1 + (256L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (256L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 28L)) + (7168L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 28L)) + (200704L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(192L + x1 + (256L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))))))) + (256L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L)))))) >= 0L) ? 0L : 28L)) + (7168L*(std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))))))) + (7168L*(((std::min(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L)))))) >= 0L) ? 0L : 28L)) + (200704L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp3 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer((1L + x2), 2L))));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer((1L + x3), 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp8 ? tmp1 : tmp9;
                            auto tmp12 = tmp11 / 9;
                            auto tmp13 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x3, 2L))));
                            auto tmp14 = tmp13 < tmp6;
                            auto tmp15 = tmp4 & tmp14;
                            auto tmp16 = decltype(tmp10)(tmp10 + tmp12);
                            auto tmp17 = tmp15 ? tmp16 : tmp10;
                            auto tmp19 = tmp18 / 9;
                            auto tmp20 = c10::convert<int>(1L + (std::max(0L, c10::div_floor_integer(x2, 2L))));
                            auto tmp21 = tmp20 < tmp3;
                            auto tmp22 = tmp21 & tmp7;
                            auto tmp23 = decltype(tmp17)(tmp17 + tmp19);
                            auto tmp24 = tmp22 ? tmp23 : tmp17;
                            auto tmp26 = tmp25 / 9;
                            auto tmp27 = tmp21 & tmp14;
                            auto tmp28 = decltype(tmp24)(tmp24 + tmp26);
                            auto tmp29 = tmp27 ? tmp28 : tmp24;
                            out_ptr0[static_cast<long>(x3 + (56L*x2) + (3136L*x1) + (200704L*x0))] = tmp29;
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
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
                    tmp22.store(out_ptr4 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_65 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_67 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (802816L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(64);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (200704L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(128);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = tmp9 & tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr2[static_cast<long>((-64L) + x2 + (64L*x1) + (200704L*x0))];
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp16 = tmp1 >= tmp10;
                        auto tmp17 = static_cast<long>(192);
                        auto tmp18 = tmp1 < tmp17;
                        auto tmp19 = tmp16 & tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr3[static_cast<long>((-128L) + x2 + (64L*x1) + (200704L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp1 >= tmp17;
                        auto tmp24 = static_cast<long>(256);
                        auto tmp25 = tmp1 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr4[static_cast<long>((-602112L) + x1 + (3136L*x2) + (200704L*x0))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = tmp19 ? tmp22 : tmp28;
                        auto tmp30 = tmp12 ? tmp15 : tmp29;
                        auto tmp31 = tmp5 ? tmp8 : tmp30;
                        auto tmp32 = static_cast<float>(0.0);
                        auto tmp33 = tmp0 ? tmp32 : tmp31;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (802816L*x0))] = tmp33;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_68 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(32L + x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(32L + x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
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
                    auto tmp11 = static_cast<float>(3.985969387755102e-05);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
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
                    auto tmp11 = static_cast<float>(3.985969387755102e-05);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_72 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(32);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (32L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(64);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-32L) + x1 + (32L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(96);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-64L) + x1 + (32L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(128);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp33;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(32L + x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(32L + x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
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
                    auto tmp11 = static_cast<float>(3.985969387755102e-05);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
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
                    auto tmp11 = static_cast<float>(3.985969387755102e-05);
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
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(32);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (32L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(64);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr2[static_cast<long>((-32L) + x1 + (32L*x0))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp16 = tmp1 >= tmp10;
                    auto tmp17 = static_cast<long>(96);
                    auto tmp18 = tmp1 < tmp17;
                    auto tmp19 = tmp16 & tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-64L) + x1 + (32L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp1 >= tmp17;
                    auto tmp24 = static_cast<long>(128);
                    auto tmp25 = tmp1 < tmp24;
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp29 = tmp19 ? tmp22 : tmp28;
                    auto tmp30 = tmp12 ? tmp15 : tmp29;
                    auto tmp31 = tmp5 ? tmp8 : tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp0 ? tmp32 : tmp31;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp33;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_78 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp25.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    tmp39.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_79 = async_compile.cpp('''
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


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(96L + x1 + (128L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(56L, 2L + x3))))) + (7168L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(56L, 2L + x2))))) + (401408L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(96L + x1 + (128L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(56L, 2L + x3))))) + (7168L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(56L, 2L + x2))))) + (401408L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(96L + x1 + (128L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(56L, 2L + x3))))) + (7168L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(56L, 2L + x2))))) + (401408L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(96L + x1 + (128L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(56L, 2L + x3))))) + (7168L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(56L, 2L + x2))))) + (401408L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(96L + x1 + (128L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(56L, 2L + x3))))) + (7168L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(56L, 2L + x2))))) + (401408L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(96L + x1 + (128L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(56L, 2L + x3))))) + (7168L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(56L, 2L + x2))))) + (401408L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(96L + x1 + (128L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(56L, 2L + x3))))) + (7168L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(56L, 2L + x2))))) + (401408L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(96L + x1 + (128L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(56L, 2L + x3))))) + (7168L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(56L, 2L + x2))))) + (401408L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(96L + x1 + (128L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(56L, 2L + x3))))) + (7168L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(56L, 2L + x2))))) + (401408L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(56L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(56L, 2L + x3));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp8 ? tmp1 : tmp9;
                            auto tmp12 = tmp11 / 9;
                            auto tmp13 = c10::convert<int>(1L + (std::max(0L, (-1L) + x3)));
                            auto tmp14 = tmp13 < tmp6;
                            auto tmp15 = tmp4 & tmp14;
                            auto tmp16 = decltype(tmp10)(tmp10 + tmp12);
                            auto tmp17 = tmp15 ? tmp16 : tmp10;
                            auto tmp19 = tmp18 / 9;
                            auto tmp20 = c10::convert<int>(2L + (std::max(0L, (-1L) + x3)));
                            auto tmp21 = tmp20 < tmp6;
                            auto tmp22 = tmp4 & tmp21;
                            auto tmp23 = decltype(tmp17)(tmp17 + tmp19);
                            auto tmp24 = tmp22 ? tmp23 : tmp17;
                            auto tmp26 = tmp25 / 9;
                            auto tmp27 = c10::convert<int>(1L + (std::max(0L, (-1L) + x2)));
                            auto tmp28 = tmp27 < tmp3;
                            auto tmp29 = tmp28 & tmp7;
                            auto tmp30 = decltype(tmp24)(tmp24 + tmp26);
                            auto tmp31 = tmp29 ? tmp30 : tmp24;
                            auto tmp33 = tmp32 / 9;
                            auto tmp34 = tmp28 & tmp14;
                            auto tmp35 = decltype(tmp31)(tmp31 + tmp33);
                            auto tmp36 = tmp34 ? tmp35 : tmp31;
                            auto tmp38 = tmp37 / 9;
                            auto tmp39 = tmp28 & tmp21;
                            auto tmp40 = decltype(tmp36)(tmp36 + tmp38);
                            auto tmp41 = tmp39 ? tmp40 : tmp36;
                            auto tmp43 = tmp42 / 9;
                            auto tmp44 = c10::convert<int>(2L + (std::max(0L, (-1L) + x2)));
                            auto tmp45 = tmp44 < tmp3;
                            auto tmp46 = tmp45 & tmp7;
                            auto tmp47 = decltype(tmp41)(tmp41 + tmp43);
                            auto tmp48 = tmp46 ? tmp47 : tmp41;
                            auto tmp50 = tmp49 / 9;
                            auto tmp51 = tmp45 & tmp14;
                            auto tmp52 = decltype(tmp48)(tmp48 + tmp50);
                            auto tmp53 = tmp51 ? tmp52 : tmp48;
                            auto tmp55 = tmp54 / 9;
                            auto tmp56 = tmp45 & tmp21;
                            auto tmp57 = decltype(tmp53)(tmp53 + tmp55);
                            auto tmp58 = tmp56 ? tmp57 : tmp53;
                            out_ptr0[static_cast<long>(x3 + (56L*x2) + (3136L*x1) + (100352L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
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
                    tmp22.store(out_ptr4 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(32L + x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(32L + x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_83 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (401408L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(32);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x1) + (100352L*x0))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(64);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = tmp9 & tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr2[static_cast<long>((-32L) + x2 + (32L*x1) + (100352L*x0))];
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp16 = tmp1 >= tmp10;
                        auto tmp17 = static_cast<long>(96);
                        auto tmp18 = tmp1 < tmp17;
                        auto tmp19 = tmp16 & tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr3[static_cast<long>((-64L) + x2 + (32L*x1) + (100352L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp1 >= tmp17;
                        auto tmp24 = static_cast<long>(128);
                        auto tmp25 = tmp1 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr4[static_cast<long>((-301056L) + x1 + (3136L*x2) + (100352L*x0))];
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp29 = tmp19 ? tmp22 : tmp28;
                        auto tmp30 = tmp12 ? tmp15 : tmp29;
                        auto tmp31 = tmp5 ? tmp8 : tmp30;
                        auto tmp32 = static_cast<float>(0.0);
                        auto tmp33 = tmp0 ? tmp32 : tmp31;
                        out_ptr0[static_cast<long>(x2 + (128L*x1) + (401408L*x0))] = tmp33;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
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
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_513, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, getitem_10, convolution_2, squeeze_7, getitem_17, convolution_3, squeeze_10, getitem_24, convolution_4, squeeze_13, getitem_31, cat, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, getitem_42, convolution_8, squeeze_25, add_46, convolution_9, squeeze_28, add_52, convolution_10, squeeze_31, cat_1, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, getitem_72, convolution_13, squeeze_40, add_74, convolution_14, squeeze_43, add_80, convolution_15, squeeze_46, cat_2, convolution_16, squeeze_49, relu_15, convolution_17, squeeze_52, getitem_102, convolution_18, squeeze_55, getitem_109, convolution_19, squeeze_58, getitem_116, convolution_20, squeeze_61, getitem_123, cat_3, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, getitem_134, convolution_24, squeeze_73, add_133, convolution_25, squeeze_76, add_139, convolution_26, squeeze_79, cat_4, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, getitem_164, convolution_29, squeeze_88, add_161, convolution_30, squeeze_91, add_167, convolution_31, squeeze_94, cat_5, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, getitem_194, convolution_34, squeeze_103, add_189, convolution_35, squeeze_106, add_195, convolution_36, squeeze_109, cat_6, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, getitem_224, convolution_39, squeeze_118, getitem_231, convolution_40, squeeze_121, getitem_238, convolution_41, squeeze_124, getitem_245, cat_7, convolution_42, squeeze_127, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, getitem_256, convolution_45, squeeze_136, add_248, convolution_46, squeeze_139, add_254, convolution_47, squeeze_142, cat_8, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, getitem_286, convolution_50, squeeze_151, add_276, convolution_51, squeeze_154, add_282, convolution_52, squeeze_157, cat_9, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, getitem_316, convolution_55, squeeze_166, add_304, convolution_56, squeeze_169, add_310, convolution_57, squeeze_172, cat_10, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, getitem_346, convolution_60, squeeze_181, add_332, convolution_61, squeeze_184, add_338, convolution_62, squeeze_187, cat_11, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, getitem_376, convolution_65, squeeze_196, add_360, convolution_66, squeeze_199, add_366, convolution_67, squeeze_202, cat_12, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, getitem_406, convolution_70, squeeze_211, getitem_413, convolution_71, squeeze_214, getitem_420, convolution_72, squeeze_217, getitem_427, cat_13, convolution_73, squeeze_220, convolution_74, squeeze_223, relu_70, convolution_75, squeeze_226, getitem_438, convolution_76, squeeze_229, add_419, convolution_77, squeeze_232, add_425, convolution_78, squeeze_235, cat_14, convolution_79, squeeze_238, relu_75, convolution_80, squeeze_241, getitem_468, convolution_81, squeeze_244, add_447, convolution_82, squeeze_247, add_453, convolution_83, squeeze_250, cat_15, convolution_84, squeeze_253, view, permute_1, le, unsqueeze_342, le_1, unsqueeze_354, le_2, unsqueeze_366, le_3, unsqueeze_378, le_4, unsqueeze_390, unsqueeze_402, le_6, unsqueeze_414, le_7, unsqueeze_426, le_8, unsqueeze_438, le_9, unsqueeze_450, unsqueeze_462, unsqueeze_474, le_11, unsqueeze_486, le_12, unsqueeze_498, le_13, unsqueeze_510, le_14, unsqueeze_522, unsqueeze_534, le_16, unsqueeze_546, le_17, unsqueeze_558, le_18, unsqueeze_570, le_19, unsqueeze_582, unsqueeze_594, le_21, unsqueeze_606, le_22, unsqueeze_618, le_23, unsqueeze_630, le_24, unsqueeze_642, unsqueeze_654, le_26, unsqueeze_666, le_27, unsqueeze_678, le_28, unsqueeze_690, le_29, unsqueeze_702, unsqueeze_714, le_31, unsqueeze_726, le_32, unsqueeze_738, le_33, unsqueeze_750, le_34, unsqueeze_762, unsqueeze_774, le_36, unsqueeze_786, le_37, unsqueeze_798, le_38, unsqueeze_810, le_39, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_41, unsqueeze_858, le_42, unsqueeze_870, le_43, unsqueeze_882, le_44, unsqueeze_894, unsqueeze_906, le_46, unsqueeze_918, le_47, unsqueeze_930, le_48, unsqueeze_942, le_49, unsqueeze_954, unsqueeze_966, le_51, unsqueeze_978, le_52, unsqueeze_990, le_53, unsqueeze_1002, le_54, unsqueeze_1014, unsqueeze_1026, le_56, unsqueeze_1038, le_57, unsqueeze_1050, le_58, unsqueeze_1062, le_59, unsqueeze_1074, unsqueeze_1086, unsqueeze_1098, le_61, unsqueeze_1110, le_62, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, le_66, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, le_71, unsqueeze_1230, le_72, unsqueeze_1242, le_73, unsqueeze_1254, le_74, unsqueeze_1266, unsqueeze_1278, unsqueeze_1290, le_76, unsqueeze_1302, le_77, unsqueeze_1314, le_78, unsqueeze_1326, le_79, unsqueeze_1338, unsqueeze_1350, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (32, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_10, (32, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_13, (32, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_14, (32, ), (1, ))
    assert_size_stride(primals_16, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_22, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_25, (32, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_26, (32, ), (1, ))
    assert_size_stride(primals_28, (32, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_31, (32, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_32, (32, ), (1, ))
    assert_size_stride(primals_34, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_40, (32, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_41, (32, ), (1, ))
    assert_size_stride(primals_43, (32, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_46, (32, 4, 3, 3), (36, 1, 12, 4))
    assert_size_stride(primals_47, (32, ), (1, ))
    assert_size_stride(primals_49, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_52, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_55, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_58, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_61, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_64, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_67, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_70, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_73, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_76, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_79, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_82, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_85, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_88, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_91, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_92, (64, ), (1, ))
    assert_size_stride(primals_94, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_97, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_100, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_103, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_104, (64, ), (1, ))
    assert_size_stride(primals_106, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_107, (64, ), (1, ))
    assert_size_stride(primals_109, (64, 8, 3, 3), (72, 1, 24, 8))
    assert_size_stride(primals_110, (64, ), (1, ))
    assert_size_stride(primals_112, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_115, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_118, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_121, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_124, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_127, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_133, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_136, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_139, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_142, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_145, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_148, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_151, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_154, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_157, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_160, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_163, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_166, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_167, (128, ), (1, ))
    assert_size_stride(primals_169, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_172, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_175, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_176, (1024, ), (1, ))
    assert_size_stride(primals_178, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_181, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_184, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_187, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_190, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_193, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_196, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_197, (128, ), (1, ))
    assert_size_stride(primals_199, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_200, (128, ), (1, ))
    assert_size_stride(primals_202, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_203, (128, ), (1, ))
    assert_size_stride(primals_205, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_208, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_209, (1024, ), (1, ))
    assert_size_stride(primals_211, (256, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_214, (256, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_215, (256, ), (1, ))
    assert_size_stride(primals_217, (256, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_220, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_221, (2048, ), (1, ))
    assert_size_stride(primals_223, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_226, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_229, (256, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_232, (256, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_235, (256, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_238, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_239, (2048, ), (1, ))
    assert_size_stride(primals_241, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_244, (256, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_245, (256, ), (1, ))
    assert_size_stride(primals_247, (256, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_250, (256, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_253, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_254, (2048, ), (1, ))
    assert_size_stride(primals_513, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem_2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(getitem_3, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_1, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_4, (128, ), (1, ))
    assert_size_stride(getitem_10, (8, 32, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_2, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_7, (32, ), (1, ))
    assert_size_stride(getitem_17, (8, 32, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_3, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_10, (32, ), (1, ))
    assert_size_stride(getitem_24, (8, 32, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_4, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_13, (32, ), (1, ))
    assert_size_stride(getitem_31, (8, 32, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(cat, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_5, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_16, (256, ), (1, ))
    assert_size_stride(convolution_6, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(relu_5, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_7, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_22, (128, ), (1, ))
    assert_size_stride(getitem_42, (8, 32, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_8, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_25, (32, ), (1, ))
    assert_size_stride(add_46, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_9, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_28, (32, ), (1, ))
    assert_size_stride(add_52, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_10, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_31, (32, ), (1, ))
    assert_size_stride(cat_1, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_11, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_34, (256, ), (1, ))
    assert_size_stride(relu_10, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_12, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(getitem_72, (8, 32, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_13, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_40, (32, ), (1, ))
    assert_size_stride(add_74, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_14, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_43, (32, ), (1, ))
    assert_size_stride(add_80, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_15, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_46, (32, ), (1, ))
    assert_size_stride(cat_2, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_16, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_49, (256, ), (1, ))
    assert_size_stride(relu_15, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_17, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_52, (256, ), (1, ))
    assert_size_stride(getitem_102, (8, 64, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_18, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_55, (64, ), (1, ))
    assert_size_stride(getitem_109, (8, 64, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_19, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_58, (64, ), (1, ))
    assert_size_stride(getitem_116, (8, 64, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_20, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_61, (64, ), (1, ))
    assert_size_stride(getitem_123, (8, 64, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(cat_3, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_21, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_64, (512, ), (1, ))
    assert_size_stride(convolution_22, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_67, (512, ), (1, ))
    assert_size_stride(relu_20, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_23, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_70, (256, ), (1, ))
    assert_size_stride(getitem_134, (8, 64, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_24, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_73, (64, ), (1, ))
    assert_size_stride(add_133, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_25, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_76, (64, ), (1, ))
    assert_size_stride(add_139, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_26, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_79, (64, ), (1, ))
    assert_size_stride(cat_4, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_27, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_82, (512, ), (1, ))
    assert_size_stride(relu_25, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_28, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(getitem_164, (8, 64, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_29, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_88, (64, ), (1, ))
    assert_size_stride(add_161, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_30, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_91, (64, ), (1, ))
    assert_size_stride(add_167, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_31, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_94, (64, ), (1, ))
    assert_size_stride(cat_5, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_32, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_97, (512, ), (1, ))
    assert_size_stride(relu_30, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_33, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_100, (256, ), (1, ))
    assert_size_stride(getitem_194, (8, 64, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_34, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_103, (64, ), (1, ))
    assert_size_stride(add_189, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_35, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_106, (64, ), (1, ))
    assert_size_stride(add_195, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_36, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_109, (64, ), (1, ))
    assert_size_stride(cat_6, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_37, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_112, (512, ), (1, ))
    assert_size_stride(relu_35, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_38, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_115, (512, ), (1, ))
    assert_size_stride(getitem_224, (8, 128, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_39, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_118, (128, ), (1, ))
    assert_size_stride(getitem_231, (8, 128, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_40, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_121, (128, ), (1, ))
    assert_size_stride(getitem_238, (8, 128, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_41, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_124, (128, ), (1, ))
    assert_size_stride(getitem_245, (8, 128, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(cat_7, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_42, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_127, (1024, ), (1, ))
    assert_size_stride(convolution_43, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_130, (1024, ), (1, ))
    assert_size_stride(relu_40, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_44, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_133, (512, ), (1, ))
    assert_size_stride(getitem_256, (8, 128, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_45, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_136, (128, ), (1, ))
    assert_size_stride(add_248, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_46, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_139, (128, ), (1, ))
    assert_size_stride(add_254, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_47, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_142, (128, ), (1, ))
    assert_size_stride(cat_8, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_48, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_145, (1024, ), (1, ))
    assert_size_stride(relu_45, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_49, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_148, (512, ), (1, ))
    assert_size_stride(getitem_286, (8, 128, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_50, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_151, (128, ), (1, ))
    assert_size_stride(add_276, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_51, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_154, (128, ), (1, ))
    assert_size_stride(add_282, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_52, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_157, (128, ), (1, ))
    assert_size_stride(cat_9, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_53, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_160, (1024, ), (1, ))
    assert_size_stride(relu_50, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_54, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_163, (512, ), (1, ))
    assert_size_stride(getitem_316, (8, 128, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_55, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_166, (128, ), (1, ))
    assert_size_stride(add_304, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_56, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_169, (128, ), (1, ))
    assert_size_stride(add_310, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_57, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_172, (128, ), (1, ))
    assert_size_stride(cat_10, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_58, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_175, (1024, ), (1, ))
    assert_size_stride(relu_55, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_59, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_178, (512, ), (1, ))
    assert_size_stride(getitem_346, (8, 128, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_60, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_181, (128, ), (1, ))
    assert_size_stride(add_332, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_61, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_184, (128, ), (1, ))
    assert_size_stride(add_338, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_62, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_187, (128, ), (1, ))
    assert_size_stride(cat_11, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_63, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_190, (1024, ), (1, ))
    assert_size_stride(relu_60, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_64, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_193, (512, ), (1, ))
    assert_size_stride(getitem_376, (8, 128, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_65, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_196, (128, ), (1, ))
    assert_size_stride(add_360, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_66, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_199, (128, ), (1, ))
    assert_size_stride(add_366, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_67, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_202, (128, ), (1, ))
    assert_size_stride(cat_12, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_68, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_205, (1024, ), (1, ))
    assert_size_stride(relu_65, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_69, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_208, (1024, ), (1, ))
    assert_size_stride(getitem_406, (8, 256, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_70, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_211, (256, ), (1, ))
    assert_size_stride(getitem_413, (8, 256, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_71, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_214, (256, ), (1, ))
    assert_size_stride(getitem_420, (8, 256, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_72, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_217, (256, ), (1, ))
    assert_size_stride(getitem_427, (8, 256, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(cat_13, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_73, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_220, (2048, ), (1, ))
    assert_size_stride(convolution_74, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_223, (2048, ), (1, ))
    assert_size_stride(relu_70, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_75, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_226, (1024, ), (1, ))
    assert_size_stride(getitem_438, (8, 256, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_76, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_229, (256, ), (1, ))
    assert_size_stride(add_419, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(convolution_77, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_232, (256, ), (1, ))
    assert_size_stride(add_425, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(convolution_78, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_235, (256, ), (1, ))
    assert_size_stride(cat_14, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_79, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_238, (2048, ), (1, ))
    assert_size_stride(relu_75, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_80, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(squeeze_241, (1024, ), (1, ))
    assert_size_stride(getitem_468, (8, 256, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_81, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_244, (256, ), (1, ))
    assert_size_stride(add_447, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(convolution_82, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_247, (256, ), (1, ))
    assert_size_stride(add_453, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(convolution_83, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_250, (256, ), (1, ))
    assert_size_stride(cat_15, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_84, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_253, (2048, ), (1, ))
    assert_size_stride(view, (8, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(unsqueeze_342, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(le_1, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(unsqueeze_354, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_2, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(unsqueeze_366, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_3, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(unsqueeze_378, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_4, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(unsqueeze_390, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(le_6, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(unsqueeze_414, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_7, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(unsqueeze_426, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_8, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(unsqueeze_438, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_9, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(unsqueeze_450, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(le_11, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(unsqueeze_486, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_12, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(unsqueeze_498, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_13, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(unsqueeze_510, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_14, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(unsqueeze_522, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_16, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_546, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_17, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_558, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_18, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_570, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_19, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_582, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_594, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_21, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_606, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_22, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_618, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_23, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_630, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_24, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_642, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_26, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_666, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_27, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_678, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_28, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_690, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_29, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_702, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_31, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_726, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_32, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_738, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_33, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_750, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_34, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_762, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_774, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_36, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_786, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_37, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_798, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_38, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_810, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_39, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(unsqueeze_822, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_846, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(le_41, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_858, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_42, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_870, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_43, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(unsqueeze_882, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_44, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(unsqueeze_894, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_906, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_46, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_918, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_47, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_930, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_48, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_942, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_49, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_954, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_966, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_51, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_978, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_52, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_990, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_53, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_1002, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_54, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1014, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1026, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_56, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_1038, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_57, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_1050, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_58, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_1062, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_59, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(unsqueeze_1074, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1086, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_1098, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(le_61, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_1110, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_62, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_1122, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_63, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(unsqueeze_1134, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_64, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(unsqueeze_1146, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1158, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_66, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(unsqueeze_1170, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_67, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(unsqueeze_1182, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_68, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(unsqueeze_1194, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_69, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(unsqueeze_1206, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1218, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_71, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(unsqueeze_1230, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_72, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(unsqueeze_1242, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_73, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(unsqueeze_1254, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_74, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(unsqueeze_1266, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1278, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_1290, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(le_76, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(unsqueeze_1302, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_77, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(unsqueeze_1314, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_78, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(unsqueeze_1326, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_79, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(unsqueeze_1338, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1350, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view, out=buf1)
    del view
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf4 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf5 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_84.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(squeeze_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del convolution_84
    del primals_254
    del squeeze_253
    del tangents_1
    del unsqueeze_342
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, cat_15, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_15
    del primals_253
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((256, ), device='cpu', dtype=torch.float32)
    buf11 = empty((256, ), device='cpu', dtype=torch.float32)
    buf12 = empty((256, ), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1(c_void_p(le_1.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_83.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del convolution_83
    del le_1
    del primals_251
    del squeeze_250
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf14 = aten.convolution_backward(buf13, add_453, primals_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_453
    del primals_250
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = buf11; del buf11  # reuse
    buf18 = empty((256, ), device='cpu', dtype=torch.float32)
    buf19 = buf13; del buf13  # reuse
    buf20 = buf18; del buf18  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_2(c_void_p(buf20.data_ptr()), c_void_p(le_2.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(convolution_82.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()))
    del convolution_82
    del le_2
    del primals_248
    del squeeze_247
    del unsqueeze_366
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf21 = aten.convolution_backward(buf19, add_447, primals_247, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_447
    del primals_247
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    buf24 = empty((256, ), device='cpu', dtype=torch.float32)
    buf25 = empty((256, ), device='cpu', dtype=torch.float32)
    buf26 = buf19; del buf19  # reuse
    buf27 = buf25; del buf25  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_3(c_void_p(buf27.data_ptr()), c_void_p(le_3.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(convolution_81.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()))
    del convolution_81
    del le_3
    del primals_245
    del squeeze_244
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf28 = aten.convolution_backward(buf26, getitem_468, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf26
    del getitem_468
    del primals_244
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf31 = buf8; del buf8  # reuse
    buf32 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf33 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf34 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf35 = buf31; del buf31  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf35.data_ptr()), c_void_p(le_4.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del buf15
    del buf22
    del convolution_80
    del le_4
    del primals_242
    del squeeze_241
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf36 = aten.convolution_backward(buf35, relu_75, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf35
    del primals_241
    buf37 = buf36[0]
    buf38 = buf36[1]
    del buf36
    buf39 = buf4; del buf4  # reuse
    buf40 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf41 = buf6; del buf6  # reuse
    buf42 = buf40; del buf40  # reuse
    buf43 = buf41; del buf41  # reuse
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_5(c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(relu_75.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(buf39.data_ptr()))
    del convolution_79
    del primals_239
    del squeeze_238
    del unsqueeze_402
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf44 = aten.convolution_backward(buf43, cat_14, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_14
    del primals_238
    buf45 = buf44[0]
    buf46 = buf44[1]
    del buf44
    buf47 = empty((256, ), device='cpu', dtype=torch.float32)
    buf48 = empty((256, ), device='cpu', dtype=torch.float32)
    buf49 = empty((256, ), device='cpu', dtype=torch.float32)
    buf50 = buf29; del buf29  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(le_6.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(convolution_78.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(squeeze_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    del convolution_78
    del le_6
    del primals_236
    del squeeze_235
    del unsqueeze_414
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf51 = aten.convolution_backward(buf50, add_425, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_425
    del primals_235
    buf52 = buf51[0]
    buf53 = buf51[1]
    del buf51
    buf54 = buf48; del buf48  # reuse
    buf55 = empty((256, ), device='cpu', dtype=torch.float32)
    buf56 = buf50; del buf50  # reuse
    buf57 = buf55; del buf55  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_7(c_void_p(buf57.data_ptr()), c_void_p(le_7.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()))
    del convolution_77
    del le_7
    del primals_233
    del squeeze_232
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf58 = aten.convolution_backward(buf56, add_419, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_419
    del primals_232
    buf59 = buf58[0]
    buf60 = buf58[1]
    del buf58
    buf61 = empty((256, ), device='cpu', dtype=torch.float32)
    buf62 = empty((256, ), device='cpu', dtype=torch.float32)
    buf63 = buf56; del buf56  # reuse
    buf64 = buf62; del buf62  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_8(c_void_p(buf64.data_ptr()), c_void_p(le_8.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf63.data_ptr()))
    del convolution_76
    del le_8
    del primals_230
    del squeeze_229
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf65 = aten.convolution_backward(buf63, getitem_438, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf63
    del getitem_438
    del primals_229
    buf66 = buf65[0]
    buf67 = buf65[1]
    del buf65
    buf68 = buf45; del buf45  # reuse
    buf69 = buf33; del buf33  # reuse
    buf70 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf71 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf72 = buf68; del buf68  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf72.data_ptr()), c_void_p(le_9.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    del buf52
    del buf59
    del convolution_75
    del le_9
    del primals_227
    del squeeze_226
    del unsqueeze_450
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf73 = aten.convolution_backward(buf72, relu_70, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_226
    buf74 = buf73[0]
    buf75 = buf73[1]
    del buf73
    buf76 = buf37; del buf37  # reuse
    buf77 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf78 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf84 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf79 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf80 = buf43; del buf43  # reuse
    buf86 = empty_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_10(c_void_p(buf76.data_ptr()), c_void_p(relu_70.data_ptr()), c_void_p(relu_75.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(squeeze_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf86.data_ptr()))
    del buf0
    del buf74
    del buf76
    del buf78
    del convolution_73
    del convolution_74
    del le
    del primals_221
    del primals_224
    del relu_70
    del relu_75
    del squeeze_223
    del unsqueeze_462
    del unsqueeze_474
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf81 = aten.convolution_backward(buf80, relu_65, primals_223, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf80
    del primals_223
    buf82 = buf81[0]
    buf83 = buf81[1]
    del buf81
    buf85 = buf84; del buf84  # reuse
    cpp_fused_native_batch_norm_backward_11(c_void_p(buf85.data_ptr()), c_void_p(squeeze_220.data_ptr()))
    del squeeze_220
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf87 = aten.convolution_backward(buf86, cat_13, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf86
    del cat_13
    del primals_220
    buf88 = buf87[0]
    buf89 = buf87[1]
    del buf87
    buf90 = reinterpret_tensor(buf72, (8, 256, 14, 14), (50176, 196, 14, 1), 0); del buf72  # reuse
    buf91 = empty((256, ), device='cpu', dtype=torch.float32)
    buf92 = empty((256, ), device='cpu', dtype=torch.float32)
    buf93 = empty((256, ), device='cpu', dtype=torch.float32)
    buf94 = buf66; del buf66  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf88.data_ptr()), c_void_p(le_11.data_ptr()), c_void_p(convolution_72.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(squeeze_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del convolution_72
    del le_11
    del primals_218
    del squeeze_217
    del unsqueeze_486
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf95 = aten.convolution_backward(buf94, getitem_420, primals_217, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del getitem_420
    del primals_217
    buf96 = buf95[0]
    buf97 = buf95[1]
    del buf95
    buf98 = buf92; del buf92  # reuse
    buf99 = empty((256, ), device='cpu', dtype=torch.float32)
    buf100 = empty((256, ), device='cpu', dtype=torch.float32)
    buf101 = buf94; del buf94  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(le_12.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del convolution_71
    del le_12
    del primals_215
    del squeeze_214
    del unsqueeze_498
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf102 = aten.convolution_backward(buf101, getitem_413, primals_214, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del getitem_413
    del primals_214
    buf103 = buf102[0]
    buf104 = buf102[1]
    del buf102
    buf105 = buf99; del buf99  # reuse
    buf106 = empty((256, ), device='cpu', dtype=torch.float32)
    buf107 = empty((256, ), device='cpu', dtype=torch.float32)
    buf108 = buf101; del buf101  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(le_13.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(squeeze_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()))
    del buf88
    del convolution_70
    del le_13
    del primals_212
    del squeeze_211
    del unsqueeze_510
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf109 = aten.convolution_backward(buf108, getitem_406, primals_211, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf108
    del getitem_406
    del primals_211
    buf110 = buf109[0]
    buf111 = buf109[1]
    del buf109
    buf112 = empty_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    buf113 = buf70; del buf70  # reuse
    buf114 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf115 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf116 = buf112; del buf112  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf116.data_ptr()), c_void_p(le_14.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(unsqueeze_522.data_ptr()), c_void_p(squeeze_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    del buf103
    del buf110
    del buf90
    del convolution_69
    del le_14
    del primals_209
    del squeeze_208
    del unsqueeze_522
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf117 = aten.convolution_backward(buf116, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_208
    buf118 = buf117[0]
    buf119 = buf117[1]
    del buf117
    buf120 = buf114; del buf114  # reuse
    buf121 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf122 = buf116; del buf116  # reuse
    buf123 = buf121; del buf121  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_16(c_void_p(buf123.data_ptr()), c_void_p(relu_65.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(unsqueeze_534.data_ptr()), c_void_p(squeeze_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()))
    del convolution_68
    del primals_206
    del squeeze_205
    del unsqueeze_534
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf124 = aten.convolution_backward(buf122, cat_12, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_12
    del primals_205
    buf125 = buf124[0]
    buf126 = buf124[1]
    del buf124
    buf127 = empty((128, ), device='cpu', dtype=torch.float32)
    buf128 = empty((128, ), device='cpu', dtype=torch.float32)
    buf129 = empty((128, ), device='cpu', dtype=torch.float32)
    buf130 = empty_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(le_16.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(squeeze_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    del convolution_67
    del le_16
    del primals_203
    del squeeze_202
    del unsqueeze_546
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf131 = aten.convolution_backward(buf130, add_366, primals_202, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_366
    del primals_202
    buf132 = buf131[0]
    buf133 = buf131[1]
    del buf131
    buf134 = buf128; del buf128  # reuse
    buf135 = empty((128, ), device='cpu', dtype=torch.float32)
    buf136 = buf130; del buf130  # reuse
    buf137 = buf135; del buf135  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_18(c_void_p(buf137.data_ptr()), c_void_p(le_17.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(unsqueeze_558.data_ptr()), c_void_p(squeeze_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    del convolution_66
    del le_17
    del primals_200
    del squeeze_199
    del unsqueeze_558
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf138 = aten.convolution_backward(buf136, add_360, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_360
    del primals_199
    buf139 = buf138[0]
    buf140 = buf138[1]
    del buf138
    buf141 = empty((128, ), device='cpu', dtype=torch.float32)
    buf142 = empty((128, ), device='cpu', dtype=torch.float32)
    buf143 = buf136; del buf136  # reuse
    buf144 = buf142; del buf142  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_19(c_void_p(buf144.data_ptr()), c_void_p(le_18.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_570.data_ptr()), c_void_p(squeeze_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()))
    del convolution_65
    del le_18
    del primals_197
    del squeeze_196
    del unsqueeze_570
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf145 = aten.convolution_backward(buf143, getitem_376, primals_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf143
    del getitem_376
    del primals_196
    buf146 = buf145[0]
    buf147 = buf145[1]
    del buf145
    buf148 = buf125; del buf125  # reuse
    buf149 = empty((512, ), device='cpu', dtype=torch.float32)
    buf150 = empty((512, ), device='cpu', dtype=torch.float32)
    buf151 = empty((512, ), device='cpu', dtype=torch.float32)
    buf152 = buf148; del buf148  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(buf152.data_ptr()), c_void_p(le_19.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(squeeze_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    del buf132
    del buf139
    del convolution_64
    del le_19
    del primals_194
    del squeeze_193
    del unsqueeze_582
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf153 = aten.convolution_backward(buf152, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf152
    del primals_193
    buf154 = buf153[0]
    buf155 = buf153[1]
    del buf153
    buf156 = buf118; del buf118  # reuse
    buf157 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf158 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf159 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf160 = buf122; del buf122  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_21(c_void_p(buf156.data_ptr()), c_void_p(relu_60.data_ptr()), c_void_p(relu_65.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(unsqueeze_594.data_ptr()), c_void_p(squeeze_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del buf154
    del buf82
    del convolution_63
    del primals_191
    del relu_60
    del relu_65
    del squeeze_190
    del unsqueeze_594
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf161 = aten.convolution_backward(buf160, cat_11, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_11
    del primals_190
    buf162 = buf161[0]
    buf163 = buf161[1]
    del buf161
    buf164 = empty((128, ), device='cpu', dtype=torch.float32)
    buf165 = empty((128, ), device='cpu', dtype=torch.float32)
    buf166 = empty((128, ), device='cpu', dtype=torch.float32)
    buf167 = buf146; del buf146  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(le_21.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(unsqueeze_606.data_ptr()), c_void_p(squeeze_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    del convolution_62
    del le_21
    del primals_188
    del squeeze_187
    del unsqueeze_606
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf168 = aten.convolution_backward(buf167, add_338, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_338
    del primals_187
    buf169 = buf168[0]
    buf170 = buf168[1]
    del buf168
    buf171 = buf165; del buf165  # reuse
    buf172 = empty((128, ), device='cpu', dtype=torch.float32)
    buf173 = buf167; del buf167  # reuse
    buf174 = buf172; del buf172  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_23(c_void_p(buf174.data_ptr()), c_void_p(le_22.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()))
    del convolution_61
    del le_22
    del primals_185
    del squeeze_184
    del unsqueeze_618
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf175 = aten.convolution_backward(buf173, add_332, primals_184, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_332
    del primals_184
    buf176 = buf175[0]
    buf177 = buf175[1]
    del buf175
    buf178 = empty((128, ), device='cpu', dtype=torch.float32)
    buf179 = empty((128, ), device='cpu', dtype=torch.float32)
    buf180 = buf173; del buf173  # reuse
    buf181 = buf179; del buf179  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_24(c_void_p(buf181.data_ptr()), c_void_p(le_23.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_630.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    del convolution_60
    del le_23
    del primals_182
    del squeeze_181
    del unsqueeze_630
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf182 = aten.convolution_backward(buf180, getitem_346, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf180
    del getitem_346
    del primals_181
    buf183 = buf182[0]
    buf184 = buf182[1]
    del buf182
    buf185 = buf162; del buf162  # reuse
    buf186 = buf150; del buf150  # reuse
    buf187 = empty((512, ), device='cpu', dtype=torch.float32)
    buf188 = empty((512, ), device='cpu', dtype=torch.float32)
    buf189 = buf185; del buf185  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_25(c_void_p(buf189.data_ptr()), c_void_p(le_24.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_642.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    del buf169
    del buf176
    del convolution_59
    del le_24
    del primals_179
    del squeeze_178
    del unsqueeze_642
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf190 = aten.convolution_backward(buf189, relu_55, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf189
    del primals_178
    buf191 = buf190[0]
    buf192 = buf190[1]
    del buf190
    buf193 = buf158; del buf158  # reuse
    buf194 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf195 = buf160; del buf160  # reuse
    buf196 = buf194; del buf194  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_26(c_void_p(buf196.data_ptr()), c_void_p(relu_55.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()))
    del convolution_58
    del primals_176
    del squeeze_175
    del unsqueeze_654
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf197 = aten.convolution_backward(buf195, cat_10, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_10
    del primals_175
    buf198 = buf197[0]
    buf199 = buf197[1]
    del buf197
    buf200 = empty((128, ), device='cpu', dtype=torch.float32)
    buf201 = empty((128, ), device='cpu', dtype=torch.float32)
    buf202 = empty((128, ), device='cpu', dtype=torch.float32)
    buf203 = buf183; del buf183  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(le_26.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(unsqueeze_666.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    del convolution_57
    del le_26
    del primals_173
    del squeeze_172
    del unsqueeze_666
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf204 = aten.convolution_backward(buf203, add_310, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_310
    del primals_172
    buf205 = buf204[0]
    buf206 = buf204[1]
    del buf204
    buf207 = buf201; del buf201  # reuse
    buf208 = empty((128, ), device='cpu', dtype=torch.float32)
    buf209 = buf203; del buf203  # reuse
    buf210 = buf208; del buf208  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_28(c_void_p(buf210.data_ptr()), c_void_p(le_27.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_678.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf209.data_ptr()))
    del convolution_56
    del le_27
    del primals_170
    del squeeze_169
    del unsqueeze_678
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf211 = aten.convolution_backward(buf209, add_304, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_304
    del primals_169
    buf212 = buf211[0]
    buf213 = buf211[1]
    del buf211
    buf214 = empty((128, ), device='cpu', dtype=torch.float32)
    buf215 = empty((128, ), device='cpu', dtype=torch.float32)
    buf216 = buf209; del buf209  # reuse
    buf217 = buf215; del buf215  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_29(c_void_p(buf217.data_ptr()), c_void_p(le_28.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    del convolution_55
    del le_28
    del primals_167
    del squeeze_166
    del unsqueeze_690
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf218 = aten.convolution_backward(buf216, getitem_316, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf216
    del getitem_316
    del primals_166
    buf219 = buf218[0]
    buf220 = buf218[1]
    del buf218
    buf221 = buf198; del buf198  # reuse
    buf222 = buf187; del buf187  # reuse
    buf223 = empty((512, ), device='cpu', dtype=torch.float32)
    buf224 = empty((512, ), device='cpu', dtype=torch.float32)
    buf225 = buf221; del buf221  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_30(c_void_p(buf225.data_ptr()), c_void_p(le_29.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_702.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()))
    del buf205
    del buf212
    del convolution_54
    del le_29
    del primals_164
    del squeeze_163
    del unsqueeze_702
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf226 = aten.convolution_backward(buf225, relu_50, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf225
    del primals_163
    buf227 = buf226[0]
    buf228 = buf226[1]
    del buf226
    buf229 = buf156; del buf156  # reuse
    buf230 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf231 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf232 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf233 = buf195; del buf195  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31(c_void_p(buf229.data_ptr()), c_void_p(relu_50.data_ptr()), c_void_p(relu_55.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_714.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    del buf191
    del convolution_53
    del primals_161
    del relu_50
    del relu_55
    del squeeze_160
    del unsqueeze_714
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf234 = aten.convolution_backward(buf233, cat_9, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_9
    del primals_160
    buf235 = buf234[0]
    buf236 = buf234[1]
    del buf234
    buf237 = empty((128, ), device='cpu', dtype=torch.float32)
    buf238 = empty((128, ), device='cpu', dtype=torch.float32)
    buf239 = empty((128, ), device='cpu', dtype=torch.float32)
    buf240 = buf219; del buf219  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(le_31.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    del convolution_52
    del le_31
    del primals_158
    del squeeze_157
    del unsqueeze_726
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf241 = aten.convolution_backward(buf240, add_282, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_282
    del primals_157
    buf242 = buf241[0]
    buf243 = buf241[1]
    del buf241
    buf244 = buf238; del buf238  # reuse
    buf245 = empty((128, ), device='cpu', dtype=torch.float32)
    buf246 = buf240; del buf240  # reuse
    buf247 = buf245; del buf245  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_33(c_void_p(buf247.data_ptr()), c_void_p(le_32.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_738.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()))
    del convolution_51
    del le_32
    del primals_155
    del squeeze_154
    del unsqueeze_738
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf248 = aten.convolution_backward(buf246, add_276, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_276
    del primals_154
    buf249 = buf248[0]
    buf250 = buf248[1]
    del buf248
    buf251 = empty((128, ), device='cpu', dtype=torch.float32)
    buf252 = empty((128, ), device='cpu', dtype=torch.float32)
    buf253 = buf246; del buf246  # reuse
    buf254 = buf252; del buf252  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_34(c_void_p(buf254.data_ptr()), c_void_p(le_33.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_750.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()))
    del convolution_50
    del le_33
    del primals_152
    del squeeze_151
    del unsqueeze_750
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf255 = aten.convolution_backward(buf253, getitem_286, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf253
    del getitem_286
    del primals_151
    buf256 = buf255[0]
    buf257 = buf255[1]
    del buf255
    buf258 = buf235; del buf235  # reuse
    buf259 = buf223; del buf223  # reuse
    buf260 = empty((512, ), device='cpu', dtype=torch.float32)
    buf261 = empty((512, ), device='cpu', dtype=torch.float32)
    buf262 = buf258; del buf258  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_35(c_void_p(buf262.data_ptr()), c_void_p(le_34.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_762.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del buf242
    del buf249
    del convolution_49
    del le_34
    del primals_149
    del squeeze_148
    del unsqueeze_762
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf263 = aten.convolution_backward(buf262, relu_45, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf262
    del primals_148
    buf264 = buf263[0]
    buf265 = buf263[1]
    del buf263
    buf266 = buf231; del buf231  # reuse
    buf267 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf268 = buf233; del buf233  # reuse
    buf269 = buf267; del buf267  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_36(c_void_p(buf269.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_774.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()))
    del convolution_48
    del primals_146
    del squeeze_145
    del unsqueeze_774
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf270 = aten.convolution_backward(buf268, cat_8, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_8
    del primals_145
    buf271 = buf270[0]
    buf272 = buf270[1]
    del buf270
    buf273 = empty((128, ), device='cpu', dtype=torch.float32)
    buf274 = empty((128, ), device='cpu', dtype=torch.float32)
    buf275 = empty((128, ), device='cpu', dtype=torch.float32)
    buf276 = buf256; del buf256  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37(c_void_p(le_36.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_786.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    del convolution_47
    del le_36
    del primals_143
    del squeeze_142
    del unsqueeze_786
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf277 = aten.convolution_backward(buf276, add_254, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_254
    del primals_142
    buf278 = buf277[0]
    buf279 = buf277[1]
    del buf277
    buf280 = buf274; del buf274  # reuse
    buf281 = empty((128, ), device='cpu', dtype=torch.float32)
    buf282 = buf276; del buf276  # reuse
    buf283 = buf281; del buf281  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_38(c_void_p(buf283.data_ptr()), c_void_p(le_37.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_798.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf282.data_ptr()))
    del convolution_46
    del le_37
    del primals_140
    del squeeze_139
    del unsqueeze_798
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf284 = aten.convolution_backward(buf282, add_248, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_248
    del primals_139
    buf285 = buf284[0]
    buf286 = buf284[1]
    del buf284
    buf287 = empty((128, ), device='cpu', dtype=torch.float32)
    buf288 = empty((128, ), device='cpu', dtype=torch.float32)
    buf289 = buf282; del buf282  # reuse
    buf290 = buf288; del buf288  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_39(c_void_p(buf290.data_ptr()), c_void_p(le_38.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_810.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()))
    del convolution_45
    del le_38
    del primals_137
    del squeeze_136
    del unsqueeze_810
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf291 = aten.convolution_backward(buf289, getitem_256, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf289
    del getitem_256
    del primals_136
    buf292 = buf291[0]
    buf293 = buf291[1]
    del buf291
    buf294 = buf271; del buf271  # reuse
    buf295 = buf260; del buf260  # reuse
    buf296 = empty((512, ), device='cpu', dtype=torch.float32)
    buf297 = empty((512, ), device='cpu', dtype=torch.float32)
    buf298 = buf294; del buf294  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_40(c_void_p(buf298.data_ptr()), c_void_p(le_39.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_822.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    del buf278
    del buf285
    del convolution_44
    del le_39
    del primals_134
    del squeeze_133
    del unsqueeze_822
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf299 = aten.convolution_backward(buf298, relu_40, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_133
    buf300 = buf299[0]
    buf301 = buf299[1]
    del buf299
    buf302 = buf229; del buf229  # reuse
    buf303 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf304 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf310 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf305 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf306 = buf268; del buf268  # reuse
    buf312 = buf227; del buf227  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_41(c_void_p(buf302.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_834.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_846.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf312.data_ptr()))
    del buf264
    del buf300
    del buf302
    del buf304
    del convolution_42
    del convolution_43
    del primals_128
    del primals_131
    del relu_40
    del relu_45
    del squeeze_130
    del unsqueeze_834
    del unsqueeze_846
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf307 = aten.convolution_backward(buf306, relu_35, primals_130, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf306
    del primals_130
    buf308 = buf307[0]
    buf309 = buf307[1]
    del buf307
    buf311 = buf310; del buf310  # reuse
    cpp_fused_native_batch_norm_backward_42(c_void_p(buf311.data_ptr()), c_void_p(squeeze_127.data_ptr()))
    del squeeze_127
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf313 = aten.convolution_backward(buf312, cat_7, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf312
    del cat_7
    del primals_127
    buf314 = buf313[0]
    buf315 = buf313[1]
    del buf313
    buf316 = reinterpret_tensor(buf298, (8, 128, 28, 28), (100352, 784, 28, 1), 0); del buf298  # reuse
    buf317 = empty((128, ), device='cpu', dtype=torch.float32)
    buf318 = empty((128, ), device='cpu', dtype=torch.float32)
    buf319 = empty((128, ), device='cpu', dtype=torch.float32)
    buf320 = buf292; del buf292  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(buf314.data_ptr()), c_void_p(le_41.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_858.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del convolution_41
    del le_41
    del primals_125
    del squeeze_124
    del unsqueeze_858
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf321 = aten.convolution_backward(buf320, getitem_238, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del getitem_238
    del primals_124
    buf322 = buf321[0]
    buf323 = buf321[1]
    del buf321
    buf324 = buf318; del buf318  # reuse
    buf325 = empty((128, ), device='cpu', dtype=torch.float32)
    buf326 = empty((128, ), device='cpu', dtype=torch.float32)
    buf327 = buf320; del buf320  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_44(c_void_p(le_42.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_870.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    del convolution_40
    del le_42
    del primals_122
    del squeeze_121
    del unsqueeze_870
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf328 = aten.convolution_backward(buf327, getitem_231, primals_121, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del getitem_231
    del primals_121
    buf329 = buf328[0]
    buf330 = buf328[1]
    del buf328
    buf331 = buf325; del buf325  # reuse
    buf332 = empty((128, ), device='cpu', dtype=torch.float32)
    buf333 = empty((128, ), device='cpu', dtype=torch.float32)
    buf334 = buf327; del buf327  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45(c_void_p(le_43.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_882.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()))
    del buf314
    del convolution_39
    del le_43
    del primals_119
    del squeeze_118
    del unsqueeze_882
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf335 = aten.convolution_backward(buf334, getitem_224, primals_118, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf334
    del getitem_224
    del primals_118
    buf336 = buf335[0]
    buf337 = buf335[1]
    del buf335
    buf338 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    buf339 = buf296; del buf296  # reuse
    buf340 = empty((512, ), device='cpu', dtype=torch.float32)
    buf341 = empty((512, ), device='cpu', dtype=torch.float32)
    buf342 = buf338; del buf338  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_46(c_void_p(buf342.data_ptr()), c_void_p(le_44.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_894.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()))
    del buf316
    del buf322
    del buf329
    del convolution_38
    del le_44
    del primals_116
    del squeeze_115
    del unsqueeze_894
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf343 = aten.convolution_backward(buf342, relu_35, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_115
    buf344 = buf343[0]
    buf345 = buf343[1]
    del buf343
    buf346 = buf340; del buf340  # reuse
    buf347 = empty((512, ), device='cpu', dtype=torch.float32)
    buf348 = buf342; del buf342  # reuse
    buf349 = buf347; del buf347  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_47(c_void_p(buf349.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_906.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()))
    del convolution_37
    del primals_113
    del squeeze_112
    del unsqueeze_906
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf350 = aten.convolution_backward(buf348, cat_6, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_6
    del primals_112
    buf351 = buf350[0]
    buf352 = buf350[1]
    del buf350
    buf353 = empty((64, ), device='cpu', dtype=torch.float32)
    buf354 = empty((64, ), device='cpu', dtype=torch.float32)
    buf355 = empty((64, ), device='cpu', dtype=torch.float32)
    buf356 = reinterpret_tensor(buf96, (8, 64, 28, 28), (50176, 1, 1792, 64), 0); del buf96  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48(c_void_p(le_46.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_918.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()))
    del convolution_36
    del le_46
    del primals_110
    del squeeze_109
    del unsqueeze_918
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf357 = aten.convolution_backward(buf356, add_195, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_195
    del primals_109
    buf358 = buf357[0]
    buf359 = buf357[1]
    del buf357
    buf360 = buf354; del buf354  # reuse
    buf361 = empty((64, ), device='cpu', dtype=torch.float32)
    buf362 = buf356; del buf356  # reuse
    buf363 = buf361; del buf361  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_49(c_void_p(buf363.data_ptr()), c_void_p(le_47.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_930.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()))
    del convolution_35
    del le_47
    del primals_107
    del squeeze_106
    del unsqueeze_930
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf364 = aten.convolution_backward(buf362, add_189, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_189
    del primals_106
    buf365 = buf364[0]
    buf366 = buf364[1]
    del buf364
    buf367 = empty((64, ), device='cpu', dtype=torch.float32)
    buf368 = empty((64, ), device='cpu', dtype=torch.float32)
    buf369 = buf362; del buf362  # reuse
    buf370 = buf368; del buf368  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_50(c_void_p(buf370.data_ptr()), c_void_p(le_48.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_942.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf369.data_ptr()))
    del convolution_34
    del le_48
    del primals_104
    del squeeze_103
    del unsqueeze_942
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf371 = aten.convolution_backward(buf369, getitem_194, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf369
    del getitem_194
    del primals_103
    buf372 = buf371[0]
    buf373 = buf371[1]
    del buf371
    buf374 = buf351; del buf351  # reuse
    buf375 = buf106; del buf106  # reuse
    buf376 = empty((256, ), device='cpu', dtype=torch.float32)
    buf377 = empty((256, ), device='cpu', dtype=torch.float32)
    buf378 = buf374; del buf374  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_51(c_void_p(buf378.data_ptr()), c_void_p(le_49.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_954.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del buf358
    del buf365
    del convolution_33
    del le_49
    del primals_101
    del squeeze_100
    del unsqueeze_954
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf379 = aten.convolution_backward(buf378, relu_30, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf378
    del primals_100
    buf380 = buf379[0]
    buf381 = buf379[1]
    del buf379
    buf382 = buf308; del buf308  # reuse
    buf383 = empty((512, ), device='cpu', dtype=torch.float32)
    buf384 = empty((512, ), device='cpu', dtype=torch.float32)
    buf385 = empty((512, ), device='cpu', dtype=torch.float32)
    buf386 = buf348; del buf348  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_52(c_void_p(buf382.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_966.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    del buf344
    del convolution_32
    del primals_98
    del relu_30
    del relu_35
    del squeeze_97
    del unsqueeze_966
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf387 = aten.convolution_backward(buf386, cat_5, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_5
    del primals_97
    buf388 = buf387[0]
    buf389 = buf387[1]
    del buf387
    buf390 = empty((64, ), device='cpu', dtype=torch.float32)
    buf391 = empty((64, ), device='cpu', dtype=torch.float32)
    buf392 = empty((64, ), device='cpu', dtype=torch.float32)
    buf393 = buf372; del buf372  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53(c_void_p(le_51.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_978.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()))
    del convolution_31
    del le_51
    del primals_95
    del squeeze_94
    del unsqueeze_978
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf394 = aten.convolution_backward(buf393, add_167, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_167
    del primals_94
    buf395 = buf394[0]
    buf396 = buf394[1]
    del buf394
    buf397 = buf391; del buf391  # reuse
    buf398 = empty((64, ), device='cpu', dtype=torch.float32)
    buf399 = buf393; del buf393  # reuse
    buf400 = buf398; del buf398  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_54(c_void_p(buf400.data_ptr()), c_void_p(le_52.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_990.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf399.data_ptr()))
    del convolution_30
    del le_52
    del primals_92
    del squeeze_91
    del unsqueeze_990
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf401 = aten.convolution_backward(buf399, add_161, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_161
    del primals_91
    buf402 = buf401[0]
    buf403 = buf401[1]
    del buf401
    buf404 = empty((64, ), device='cpu', dtype=torch.float32)
    buf405 = empty((64, ), device='cpu', dtype=torch.float32)
    buf406 = buf399; del buf399  # reuse
    buf407 = buf405; del buf405  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_55(c_void_p(buf407.data_ptr()), c_void_p(le_53.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_1002.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf406.data_ptr()))
    del convolution_29
    del le_53
    del primals_89
    del squeeze_88
    del unsqueeze_1002
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf408 = aten.convolution_backward(buf406, getitem_164, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf406
    del getitem_164
    del primals_88
    buf409 = buf408[0]
    buf410 = buf408[1]
    del buf408
    buf411 = buf388; del buf388  # reuse
    buf412 = buf376; del buf376  # reuse
    buf413 = empty((256, ), device='cpu', dtype=torch.float32)
    buf414 = empty((256, ), device='cpu', dtype=torch.float32)
    buf415 = buf411; del buf411  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_56(c_void_p(buf415.data_ptr()), c_void_p(le_54.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_1014.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()))
    del buf395
    del buf402
    del convolution_28
    del le_54
    del primals_86
    del squeeze_85
    del unsqueeze_1014
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf416 = aten.convolution_backward(buf415, relu_25, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf415
    del primals_85
    buf417 = buf416[0]
    buf418 = buf416[1]
    del buf416
    buf419 = buf384; del buf384  # reuse
    buf420 = empty((512, ), device='cpu', dtype=torch.float32)
    buf421 = buf386; del buf386  # reuse
    buf422 = buf420; del buf420  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_57(c_void_p(buf422.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_1026.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()))
    del convolution_27
    del primals_83
    del squeeze_82
    del unsqueeze_1026
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf423 = aten.convolution_backward(buf421, cat_4, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_4
    del primals_82
    buf424 = buf423[0]
    buf425 = buf423[1]
    del buf423
    buf426 = empty((64, ), device='cpu', dtype=torch.float32)
    buf427 = empty((64, ), device='cpu', dtype=torch.float32)
    buf428 = empty((64, ), device='cpu', dtype=torch.float32)
    buf429 = buf409; del buf409  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58(c_void_p(le_56.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_1038.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    del convolution_26
    del le_56
    del primals_80
    del squeeze_79
    del unsqueeze_1038
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf430 = aten.convolution_backward(buf429, add_139, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_139
    del primals_79
    buf431 = buf430[0]
    buf432 = buf430[1]
    del buf430
    buf433 = buf427; del buf427  # reuse
    buf434 = empty((64, ), device='cpu', dtype=torch.float32)
    buf435 = buf429; del buf429  # reuse
    buf436 = buf434; del buf434  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_59(c_void_p(buf436.data_ptr()), c_void_p(le_57.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_1050.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf435.data_ptr()))
    del convolution_25
    del le_57
    del primals_77
    del squeeze_76
    del unsqueeze_1050
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf437 = aten.convolution_backward(buf435, add_133, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_133
    del primals_76
    buf438 = buf437[0]
    buf439 = buf437[1]
    del buf437
    buf440 = empty((64, ), device='cpu', dtype=torch.float32)
    buf441 = empty((64, ), device='cpu', dtype=torch.float32)
    buf442 = buf435; del buf435  # reuse
    buf443 = buf441; del buf441  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_60(c_void_p(buf443.data_ptr()), c_void_p(le_58.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_1062.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf442.data_ptr()))
    del convolution_24
    del le_58
    del primals_74
    del squeeze_73
    del unsqueeze_1062
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf444 = aten.convolution_backward(buf442, getitem_134, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf442
    del getitem_134
    del primals_73
    buf445 = buf444[0]
    buf446 = buf444[1]
    del buf444
    buf447 = buf424; del buf424  # reuse
    buf448 = buf413; del buf413  # reuse
    buf449 = empty((256, ), device='cpu', dtype=torch.float32)
    buf450 = empty((256, ), device='cpu', dtype=torch.float32)
    buf451 = buf447; del buf447  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_61(c_void_p(buf451.data_ptr()), c_void_p(le_59.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_1074.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()))
    del buf431
    del buf438
    del convolution_23
    del le_59
    del primals_71
    del squeeze_70
    del unsqueeze_1074
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf452 = aten.convolution_backward(buf451, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_70
    buf453 = buf452[0]
    buf454 = buf452[1]
    del buf452
    buf455 = buf382; del buf382  # reuse
    buf456 = empty((512, ), device='cpu', dtype=torch.float32)
    buf457 = empty((512, ), device='cpu', dtype=torch.float32)
    buf463 = empty((512, ), device='cpu', dtype=torch.float32)
    buf458 = empty((512, ), device='cpu', dtype=torch.float32)
    buf459 = buf421; del buf421  # reuse
    buf465 = buf380; del buf380  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_62(c_void_p(buf455.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_1086.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_1098.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf465.data_ptr()))
    del buf417
    del buf453
    del buf455
    del buf457
    del convolution_21
    del convolution_22
    del primals_65
    del primals_68
    del relu_20
    del relu_25
    del squeeze_67
    del unsqueeze_1086
    del unsqueeze_1098
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf460 = aten.convolution_backward(buf459, relu_15, primals_67, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf459
    del primals_67
    buf461 = buf460[0]
    buf462 = buf460[1]
    del buf460
    buf464 = buf463; del buf463  # reuse
    cpp_fused_native_batch_norm_backward_63(c_void_p(buf464.data_ptr()), c_void_p(squeeze_64.data_ptr()))
    del squeeze_64
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf466 = aten.convolution_backward(buf465, cat_3, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf465
    del cat_3
    del primals_64
    buf467 = buf466[0]
    buf468 = buf466[1]
    del buf466
    buf469 = reinterpret_tensor(buf451, (8, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf451  # reuse
    buf470 = empty((64, ), device='cpu', dtype=torch.float32)
    buf471 = empty((64, ), device='cpu', dtype=torch.float32)
    buf472 = empty((64, ), device='cpu', dtype=torch.float32)
    buf473 = buf445; del buf445  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_64(c_void_p(buf467.data_ptr()), c_void_p(le_61.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_1110.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()))
    del convolution_20
    del le_61
    del primals_62
    del squeeze_61
    del unsqueeze_1110
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf474 = aten.convolution_backward(buf473, getitem_116, primals_61, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del getitem_116
    del primals_61
    buf475 = buf474[0]
    buf476 = buf474[1]
    del buf474
    buf477 = buf471; del buf471  # reuse
    buf478 = empty((64, ), device='cpu', dtype=torch.float32)
    buf479 = empty((64, ), device='cpu', dtype=torch.float32)
    buf480 = buf473; del buf473  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_65(c_void_p(le_62.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_1122.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()))
    del convolution_19
    del le_62
    del primals_59
    del squeeze_58
    del unsqueeze_1122
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf481 = aten.convolution_backward(buf480, getitem_109, primals_58, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del getitem_109
    del primals_58
    buf482 = buf481[0]
    buf483 = buf481[1]
    del buf481
    buf484 = buf478; del buf478  # reuse
    buf485 = empty((64, ), device='cpu', dtype=torch.float32)
    buf486 = empty((64, ), device='cpu', dtype=torch.float32)
    buf487 = buf480; del buf480  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66(c_void_p(le_63.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_1134.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()))
    del buf467
    del convolution_18
    del le_63
    del primals_56
    del squeeze_55
    del unsqueeze_1134
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf488 = aten.convolution_backward(buf487, getitem_102, primals_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf487
    del getitem_102
    del primals_55
    buf489 = buf488[0]
    buf490 = buf488[1]
    del buf488
    buf491 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    buf492 = buf449; del buf449  # reuse
    buf493 = empty((256, ), device='cpu', dtype=torch.float32)
    buf494 = empty((256, ), device='cpu', dtype=torch.float32)
    buf495 = buf491; del buf491  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_67(c_void_p(buf495.data_ptr()), c_void_p(le_64.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_1146.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()))
    del buf469
    del buf475
    del buf482
    del buf489
    del convolution_17
    del le_64
    del primals_53
    del squeeze_52
    del unsqueeze_1146
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf496 = aten.convolution_backward(buf495, relu_15, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_52
    buf497 = buf496[0]
    buf498 = buf496[1]
    del buf496
    buf499 = buf493; del buf493  # reuse
    buf500 = empty((256, ), device='cpu', dtype=torch.float32)
    buf501 = buf495; del buf495  # reuse
    buf502 = buf500; del buf500  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_68(c_void_p(buf502.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_1158.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf501.data_ptr()))
    del convolution_16
    del primals_50
    del squeeze_49
    del unsqueeze_1158
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf503 = aten.convolution_backward(buf501, cat_2, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_2
    del primals_49
    buf504 = buf503[0]
    buf505 = buf503[1]
    del buf503
    buf506 = empty((32, ), device='cpu', dtype=torch.float32)
    buf507 = empty((32, ), device='cpu', dtype=torch.float32)
    buf508 = empty((32, ), device='cpu', dtype=torch.float32)
    buf509 = reinterpret_tensor(buf336, (8, 32, 56, 56), (100352, 1, 1792, 32), 0); del buf336  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69(c_void_p(le_66.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_1170.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()))
    del convolution_15
    del le_66
    del primals_47
    del squeeze_46
    del unsqueeze_1170
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf510 = aten.convolution_backward(buf509, add_80, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_80
    del primals_46
    buf511 = buf510[0]
    buf512 = buf510[1]
    del buf510
    buf513 = buf507; del buf507  # reuse
    buf514 = empty((32, ), device='cpu', dtype=torch.float32)
    buf515 = buf509; del buf509  # reuse
    buf516 = buf514; del buf514  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_70(c_void_p(buf516.data_ptr()), c_void_p(le_67.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_1182.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf515.data_ptr()))
    del convolution_14
    del le_67
    del primals_44
    del squeeze_43
    del unsqueeze_1182
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf517 = aten.convolution_backward(buf515, add_74, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_74
    del primals_43
    buf518 = buf517[0]
    buf519 = buf517[1]
    del buf517
    buf520 = empty((32, ), device='cpu', dtype=torch.float32)
    buf521 = empty((32, ), device='cpu', dtype=torch.float32)
    buf522 = buf515; del buf515  # reuse
    buf523 = buf521; del buf521  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_71(c_void_p(buf523.data_ptr()), c_void_p(le_68.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_1194.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf522.data_ptr()))
    del convolution_13
    del le_68
    del primals_41
    del squeeze_40
    del unsqueeze_1194
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf524 = aten.convolution_backward(buf522, getitem_72, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf522
    del getitem_72
    del primals_40
    buf525 = buf524[0]
    buf526 = buf524[1]
    del buf524
    buf527 = buf504; del buf504  # reuse
    buf528 = buf332; del buf332  # reuse
    buf529 = empty((128, ), device='cpu', dtype=torch.float32)
    buf530 = empty((128, ), device='cpu', dtype=torch.float32)
    buf531 = buf527; del buf527  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_72(c_void_p(buf531.data_ptr()), c_void_p(le_69.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_1206.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()))
    del buf511
    del buf518
    del convolution_12
    del le_69
    del primals_38
    del squeeze_37
    del unsqueeze_1206
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf532 = aten.convolution_backward(buf531, relu_10, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf531
    del primals_37
    buf533 = buf532[0]
    buf534 = buf532[1]
    del buf532
    buf535 = buf461; del buf461  # reuse
    buf536 = empty((256, ), device='cpu', dtype=torch.float32)
    buf537 = empty((256, ), device='cpu', dtype=torch.float32)
    buf538 = empty((256, ), device='cpu', dtype=torch.float32)
    buf539 = buf501; del buf501  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_73(c_void_p(buf535.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_1218.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()))
    del buf497
    del convolution_11
    del primals_35
    del relu_10
    del relu_15
    del squeeze_34
    del unsqueeze_1218
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf540 = aten.convolution_backward(buf539, cat_1, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_1
    del primals_34
    buf541 = buf540[0]
    buf542 = buf540[1]
    del buf540
    buf543 = empty((32, ), device='cpu', dtype=torch.float32)
    buf544 = empty((32, ), device='cpu', dtype=torch.float32)
    buf545 = empty((32, ), device='cpu', dtype=torch.float32)
    buf546 = buf525; del buf525  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_74(c_void_p(le_71.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_1230.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()))
    del convolution_10
    del le_71
    del primals_32
    del squeeze_31
    del unsqueeze_1230
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf547 = aten.convolution_backward(buf546, add_52, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_52
    del primals_31
    buf548 = buf547[0]
    buf549 = buf547[1]
    del buf547
    buf550 = buf544; del buf544  # reuse
    buf551 = empty((32, ), device='cpu', dtype=torch.float32)
    buf552 = buf546; del buf546  # reuse
    buf553 = buf551; del buf551  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_75(c_void_p(buf553.data_ptr()), c_void_p(le_72.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_1242.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf552.data_ptr()))
    del convolution_9
    del le_72
    del primals_29
    del squeeze_28
    del unsqueeze_1242
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf554 = aten.convolution_backward(buf552, add_46, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_46
    del primals_28
    buf555 = buf554[0]
    buf556 = buf554[1]
    del buf554
    buf557 = empty((32, ), device='cpu', dtype=torch.float32)
    buf558 = empty((32, ), device='cpu', dtype=torch.float32)
    buf559 = buf552; del buf552  # reuse
    buf560 = buf558; del buf558  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_76(c_void_p(buf560.data_ptr()), c_void_p(le_73.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_1254.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf559.data_ptr()))
    del convolution_8
    del le_73
    del primals_26
    del squeeze_25
    del unsqueeze_1254
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf561 = aten.convolution_backward(buf559, getitem_42, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf559
    del getitem_42
    del primals_25
    buf562 = buf561[0]
    buf563 = buf561[1]
    del buf561
    buf564 = buf541; del buf541  # reuse
    buf565 = buf529; del buf529  # reuse
    buf566 = empty((128, ), device='cpu', dtype=torch.float32)
    buf567 = empty((128, ), device='cpu', dtype=torch.float32)
    buf568 = buf564; del buf564  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_77(c_void_p(buf568.data_ptr()), c_void_p(le_74.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_1266.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()))
    del buf548
    del convolution_7
    del le_74
    del primals_23
    del squeeze_22
    del unsqueeze_1266
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf569 = aten.convolution_backward(buf568, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf568
    del primals_22
    buf570 = buf569[0]
    buf571 = buf569[1]
    del buf569
    buf572 = buf537; del buf537  # reuse
    buf573 = empty((256, ), device='cpu', dtype=torch.float32)
    buf579 = empty((256, ), device='cpu', dtype=torch.float32)
    buf574 = buf539; del buf539  # reuse
    buf580 = buf533; del buf533  # reuse
    buf575 = buf573; del buf573  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_78(c_void_p(buf575.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_1278.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_1290.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf580.data_ptr()))
    del buf535
    del buf570
    del convolution_5
    del convolution_6
    del primals_17
    del primals_20
    del relu_5
    del squeeze_19
    del unsqueeze_1278
    del unsqueeze_1290
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf576 = aten.convolution_backward(buf574, getitem_2, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf574
    del primals_19
    buf577 = buf576[0]
    buf578 = buf576[1]
    del buf576
    buf581 = buf579; del buf579  # reuse
    cpp_fused_native_batch_norm_backward_79(c_void_p(buf581.data_ptr()), c_void_p(squeeze_16.data_ptr()))
    del squeeze_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf582 = aten.convolution_backward(buf580, cat, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf580
    del cat
    del primals_16
    buf583 = buf582[0]
    buf584 = buf582[1]
    del buf582
    buf585 = reinterpret_tensor(buf562, (8, 32, 56, 56), (100352, 3136, 56, 1), 0); del buf562  # reuse
    buf586 = empty((32, ), device='cpu', dtype=torch.float32)
    buf587 = empty((32, ), device='cpu', dtype=torch.float32)
    buf588 = empty((32, ), device='cpu', dtype=torch.float32)
    buf589 = buf555; del buf555  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_80(c_void_p(buf583.data_ptr()), c_void_p(le_76.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_1302.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf589.data_ptr()))
    del convolution_4
    del le_76
    del primals_14
    del squeeze_13
    del unsqueeze_1302
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf590 = aten.convolution_backward(buf589, getitem_24, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del getitem_24
    del primals_13
    buf591 = buf590[0]
    buf592 = buf590[1]
    del buf590
    buf593 = buf587; del buf587  # reuse
    buf594 = empty((32, ), device='cpu', dtype=torch.float32)
    buf595 = empty((32, ), device='cpu', dtype=torch.float32)
    buf596 = buf589; del buf589  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81(c_void_p(le_77.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_1314.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()))
    del convolution_3
    del le_77
    del primals_11
    del squeeze_10
    del unsqueeze_1314
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf597 = aten.convolution_backward(buf596, getitem_17, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del getitem_17
    del primals_10
    buf598 = buf597[0]
    buf599 = buf597[1]
    del buf597
    buf600 = buf594; del buf594  # reuse
    buf601 = empty((32, ), device='cpu', dtype=torch.float32)
    buf602 = empty((32, ), device='cpu', dtype=torch.float32)
    buf603 = buf596; del buf596  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82(c_void_p(le_78.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_1326.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf603.data_ptr()))
    del buf601
    del convolution_2
    del le_78
    del primals_8
    del squeeze_7
    del unsqueeze_1326
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf604 = aten.convolution_backward(buf603, getitem_10, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf603
    del getitem_10
    del primals_7
    buf605 = buf604[0]
    buf606 = buf604[1]
    del buf604
    buf607 = buf583; del buf583  # reuse
    buf608 = buf566; del buf566  # reuse
    buf609 = empty((128, ), device='cpu', dtype=torch.float32)
    buf610 = empty((128, ), device='cpu', dtype=torch.float32)
    buf611 = buf607; del buf607  # reuse
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_threshold_backward_83(c_void_p(buf611.data_ptr()), c_void_p(le_79.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_1338.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf610.data_ptr()))
    del buf585
    del buf591
    del buf598
    del buf605
    del buf609
    del convolution_1
    del le_79
    del primals_5
    del squeeze_4
    del unsqueeze_1338
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf612 = aten.convolution_backward(buf611, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf611
    del getitem_2
    del primals_4
    buf613 = buf612[0]
    buf614 = buf612[1]
    del buf612
    buf615 = buf577; del buf577  # reuse
    cpp_fused_add_84(c_void_p(buf615.data_ptr()), c_void_p(buf613.data_ptr()))
    del buf613
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf616 = aten.max_pool2d_with_indices_backward(buf615, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3)
    del buf615
    del getitem_3
    buf617 = buf616
    del buf616
    buf618 = buf485; del buf485  # reuse
    buf619 = empty((64, ), device='cpu', dtype=torch.float32)
    buf620 = empty((64, ), device='cpu', dtype=torch.float32)
    buf621 = buf617; del buf617  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_85(c_void_p(buf621.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_1350.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()))
    del buf619
    del convolution
    del primals_2
    del relu
    del squeeze_1
    del unsqueeze_1350
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf622 = aten.convolution_backward(buf621, primals_513, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf621
    del primals_1
    del primals_513
    buf623 = buf622[1]
    return (buf623, buf620, buf618, buf614, buf610, buf608, buf606, buf602, buf600, buf599, buf595, buf593, buf592, buf588, buf586, buf584, buf581, buf572, buf578, buf575, buf572, buf571, buf567, buf565, buf563, buf560, buf557, buf556, buf553, buf550, buf549, buf545, buf543, buf542, buf538, buf536, buf534, buf530, buf528, buf526, buf523, buf520, buf519, buf516, buf513, buf512, buf508, buf506, buf505, buf502, buf499, buf498, buf494, buf492, buf490, buf486, buf484, buf483, buf479, buf477, buf476, buf472, buf470, buf468, buf464, buf456, buf462, buf458, buf456, buf454, buf450, buf448, buf446, buf443, buf440, buf439, buf436, buf433, buf432, buf428, buf426, buf425, buf422, buf419, buf418, buf414, buf412, buf410, buf407, buf404, buf403, buf400, buf397, buf396, buf392, buf390, buf389, buf385, buf383, buf381, buf377, buf375, buf373, buf370, buf367, buf366, buf363, buf360, buf359, buf355, buf353, buf352, buf349, buf346, buf345, buf341, buf339, buf337, buf333, buf331, buf330, buf326, buf324, buf323, buf319, buf317, buf315, buf311, buf303, buf309, buf305, buf303, buf301, buf297, buf295, buf293, buf290, buf287, buf286, buf283, buf280, buf279, buf275, buf273, buf272, buf269, buf266, buf265, buf261, buf259, buf257, buf254, buf251, buf250, buf247, buf244, buf243, buf239, buf237, buf236, buf232, buf230, buf228, buf224, buf222, buf220, buf217, buf214, buf213, buf210, buf207, buf206, buf202, buf200, buf199, buf196, buf193, buf192, buf188, buf186, buf184, buf181, buf178, buf177, buf174, buf171, buf170, buf166, buf164, buf163, buf159, buf157, buf155, buf151, buf149, buf147, buf144, buf141, buf140, buf137, buf134, buf133, buf129, buf127, buf126, buf123, buf120, buf119, buf115, buf113, buf111, buf107, buf105, buf104, buf100, buf98, buf97, buf93, buf91, buf89, buf85, buf77, buf83, buf79, buf77, buf75, buf71, buf69, buf67, buf64, buf61, buf60, buf57, buf54, buf53, buf49, buf47, buf46, buf42, buf39, buf38, buf34, buf32, buf30, buf27, buf24, buf23, buf20, buf17, buf16, buf12, buf10, buf9, buf5, buf3, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((32, 4, 3, 3), (36, 1, 12, 4), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((64, 8, 3, 3), (72, 1, 24, 8), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((256, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_513 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.int64)
    convolution_1 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_10 = rand_strided((8, 32, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((8, 32, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_24 = rand_strided((8, 32, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((8, 32, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_42 = rand_strided((8, 32, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_46 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_52 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_72 = rand_strided((8, 32, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_74 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_80 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_102 = rand_strided((8, 64, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_109 = rand_strided((8, 64, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_116 = rand_strided((8, 64, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_123 = rand_strided((8, 64, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_134 = rand_strided((8, 64, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_133 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_139 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_164 = rand_strided((8, 64, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_161 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_167 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    cat_5 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_194 = rand_strided((8, 64, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_189 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_195 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    cat_6 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_224 = rand_strided((8, 128, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_231 = rand_strided((8, 128, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_238 = rand_strided((8, 128, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_245 = rand_strided((8, 128, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    cat_7 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_40 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_256 = rand_strided((8, 128, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_248 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_254 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat_8 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_286 = rand_strided((8, 128, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_276 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_282 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat_9 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_50 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_316 = rand_strided((8, 128, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_304 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_310 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat_10 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_55 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_346 = rand_strided((8, 128, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_332 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_338 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_187 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat_11 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    squeeze_190 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_60 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_193 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_376 = rand_strided((8, 128, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_196 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_360 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_199 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_366 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_202 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat_12 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    squeeze_205 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    relu_65 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    squeeze_208 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_406 = rand_strided((8, 256, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_211 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_413 = rand_strided((8, 256, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_214 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_420 = rand_strided((8, 256, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_217 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_427 = rand_strided((8, 256, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    cat_13 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    squeeze_220 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    squeeze_223 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    relu_70 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    squeeze_226 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_438 = rand_strided((8, 256, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_229 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_419 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_232 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_425 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    convolution_78 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_235 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    cat_14 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    squeeze_238 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    relu_75 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    squeeze_241 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_468 = rand_strided((8, 256, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_81 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_244 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_447 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    convolution_82 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_247 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_453 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    convolution_83 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_250 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    cat_15 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_84 = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    squeeze_253 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    view = rand_strided((8, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.bool)
    unsqueeze_342 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_1 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.bool)
    unsqueeze_354 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_2 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.bool)
    unsqueeze_366 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_3 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.bool)
    unsqueeze_378 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_4 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.bool)
    unsqueeze_390 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_6 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.bool)
    unsqueeze_414 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_7 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.bool)
    unsqueeze_426 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_8 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.bool)
    unsqueeze_438 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_9 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.bool)
    unsqueeze_450 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_11 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.bool)
    unsqueeze_486 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_12 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.bool)
    unsqueeze_498 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_13 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.bool)
    unsqueeze_510 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_14 = rand_strided((8, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.bool)
    unsqueeze_522 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_16 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_546 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_17 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_18 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_570 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_19 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_582 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_21 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_606 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_22 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_618 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_23 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_630 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_24 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_642 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_26 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_666 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_27 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_678 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_28 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_690 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_29 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_702 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_31 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_726 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_32 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_738 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_33 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_750 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_34 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_762 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_36 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_786 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_37 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_798 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_38 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_810 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_39 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.bool)
    unsqueeze_822 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_41 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_858 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_42 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_870 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_43 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.bool)
    unsqueeze_882 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_44 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.bool)
    unsqueeze_894 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_906 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_46 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_918 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_47 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_930 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_48 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_942 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_49 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.bool)
    unsqueeze_954 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_966 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_51 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_978 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_52 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_990 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_53 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1002 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_54 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.bool)
    unsqueeze_1014 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1026 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_56 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1038 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_57 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1050 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_58 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1062 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_59 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.bool)
    unsqueeze_1074 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1086 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1098 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_61 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1110 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_62 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1122 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_63 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1134 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_64 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.bool)
    unsqueeze_1146 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1158 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_66 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1170 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_67 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1182 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_68 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1194 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_69 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.bool)
    unsqueeze_1206 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1218 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_71 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1230 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_72 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1242 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_73 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1254 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_74 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.bool)
    unsqueeze_1266 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1278 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1290 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_76 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1302 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_77 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1314 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_78 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1326 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_79 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.bool)
    unsqueeze_1338 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1350 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_513, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, getitem_10, convolution_2, squeeze_7, getitem_17, convolution_3, squeeze_10, getitem_24, convolution_4, squeeze_13, getitem_31, cat, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, getitem_42, convolution_8, squeeze_25, add_46, convolution_9, squeeze_28, add_52, convolution_10, squeeze_31, cat_1, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, getitem_72, convolution_13, squeeze_40, add_74, convolution_14, squeeze_43, add_80, convolution_15, squeeze_46, cat_2, convolution_16, squeeze_49, relu_15, convolution_17, squeeze_52, getitem_102, convolution_18, squeeze_55, getitem_109, convolution_19, squeeze_58, getitem_116, convolution_20, squeeze_61, getitem_123, cat_3, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, getitem_134, convolution_24, squeeze_73, add_133, convolution_25, squeeze_76, add_139, convolution_26, squeeze_79, cat_4, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, getitem_164, convolution_29, squeeze_88, add_161, convolution_30, squeeze_91, add_167, convolution_31, squeeze_94, cat_5, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, getitem_194, convolution_34, squeeze_103, add_189, convolution_35, squeeze_106, add_195, convolution_36, squeeze_109, cat_6, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, getitem_224, convolution_39, squeeze_118, getitem_231, convolution_40, squeeze_121, getitem_238, convolution_41, squeeze_124, getitem_245, cat_7, convolution_42, squeeze_127, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, getitem_256, convolution_45, squeeze_136, add_248, convolution_46, squeeze_139, add_254, convolution_47, squeeze_142, cat_8, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, getitem_286, convolution_50, squeeze_151, add_276, convolution_51, squeeze_154, add_282, convolution_52, squeeze_157, cat_9, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, getitem_316, convolution_55, squeeze_166, add_304, convolution_56, squeeze_169, add_310, convolution_57, squeeze_172, cat_10, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, getitem_346, convolution_60, squeeze_181, add_332, convolution_61, squeeze_184, add_338, convolution_62, squeeze_187, cat_11, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, getitem_376, convolution_65, squeeze_196, add_360, convolution_66, squeeze_199, add_366, convolution_67, squeeze_202, cat_12, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, getitem_406, convolution_70, squeeze_211, getitem_413, convolution_71, squeeze_214, getitem_420, convolution_72, squeeze_217, getitem_427, cat_13, convolution_73, squeeze_220, convolution_74, squeeze_223, relu_70, convolution_75, squeeze_226, getitem_438, convolution_76, squeeze_229, add_419, convolution_77, squeeze_232, add_425, convolution_78, squeeze_235, cat_14, convolution_79, squeeze_238, relu_75, convolution_80, squeeze_241, getitem_468, convolution_81, squeeze_244, add_447, convolution_82, squeeze_247, add_453, convolution_83, squeeze_250, cat_15, convolution_84, squeeze_253, view, permute_1, le, unsqueeze_342, le_1, unsqueeze_354, le_2, unsqueeze_366, le_3, unsqueeze_378, le_4, unsqueeze_390, unsqueeze_402, le_6, unsqueeze_414, le_7, unsqueeze_426, le_8, unsqueeze_438, le_9, unsqueeze_450, unsqueeze_462, unsqueeze_474, le_11, unsqueeze_486, le_12, unsqueeze_498, le_13, unsqueeze_510, le_14, unsqueeze_522, unsqueeze_534, le_16, unsqueeze_546, le_17, unsqueeze_558, le_18, unsqueeze_570, le_19, unsqueeze_582, unsqueeze_594, le_21, unsqueeze_606, le_22, unsqueeze_618, le_23, unsqueeze_630, le_24, unsqueeze_642, unsqueeze_654, le_26, unsqueeze_666, le_27, unsqueeze_678, le_28, unsqueeze_690, le_29, unsqueeze_702, unsqueeze_714, le_31, unsqueeze_726, le_32, unsqueeze_738, le_33, unsqueeze_750, le_34, unsqueeze_762, unsqueeze_774, le_36, unsqueeze_786, le_37, unsqueeze_798, le_38, unsqueeze_810, le_39, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_41, unsqueeze_858, le_42, unsqueeze_870, le_43, unsqueeze_882, le_44, unsqueeze_894, unsqueeze_906, le_46, unsqueeze_918, le_47, unsqueeze_930, le_48, unsqueeze_942, le_49, unsqueeze_954, unsqueeze_966, le_51, unsqueeze_978, le_52, unsqueeze_990, le_53, unsqueeze_1002, le_54, unsqueeze_1014, unsqueeze_1026, le_56, unsqueeze_1038, le_57, unsqueeze_1050, le_58, unsqueeze_1062, le_59, unsqueeze_1074, unsqueeze_1086, unsqueeze_1098, le_61, unsqueeze_1110, le_62, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, le_66, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, le_71, unsqueeze_1230, le_72, unsqueeze_1242, le_73, unsqueeze_1254, le_74, unsqueeze_1266, unsqueeze_1278, unsqueeze_1290, le_76, unsqueeze_1302, le_77, unsqueeze_1314, le_78, unsqueeze_1326, le_79, unsqueeze_1338, unsqueeze_1350, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2next50', benchmark_compiled_module)
