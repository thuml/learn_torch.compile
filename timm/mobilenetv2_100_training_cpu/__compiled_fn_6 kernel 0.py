
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


cpp_fused_convolution_backward_div_hardtanh_backward_native_batch_norm_backward_sum_0 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (1280L*x1) + (62720L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1280L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1280L*x1) + (62720L*x0)));
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
                        tmp25.store(out_ptr4 + static_cast<long>(x2 + (1280L*x1) + (62720L*x0)));
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
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.002551020408163265);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (160L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
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
                tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
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
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (576L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (576L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (576L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (576L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_16 = async_compile.cpp('''
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
                       float* out_ptr3)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (576L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (576L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (576L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (576L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_25 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.0006377551020408163);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
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
                tmp22.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
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
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (64L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
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
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00015943877551020407);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (144L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (144L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (144L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (144L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.985969387755102e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(9.964923469387754e-06);
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(9.964923469387754e-06);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    auto tmp9 = static_cast<float>(9.964923469387754e-06);
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    auto tmp9 = static_cast<float>(9.964923469387754e-06);
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_315, convolution, squeeze_1, clamp_max, convolution_1, squeeze_4, clamp_max_1, convolution_2, squeeze_7, add_14, convolution_3, squeeze_10, clamp_max_2, convolution_4, squeeze_13, clamp_max_3, convolution_5, squeeze_16, add_29, convolution_6, squeeze_19, clamp_max_4, convolution_7, squeeze_22, clamp_max_5, convolution_8, squeeze_25, add_45, convolution_9, squeeze_28, clamp_max_6, convolution_10, squeeze_31, clamp_max_7, convolution_11, squeeze_34, add_60, convolution_12, squeeze_37, clamp_max_8, convolution_13, squeeze_40, clamp_max_9, convolution_14, squeeze_43, add_76, convolution_15, squeeze_46, clamp_max_10, convolution_16, squeeze_49, clamp_max_11, convolution_17, squeeze_52, add_92, convolution_18, squeeze_55, clamp_max_12, convolution_19, squeeze_58, clamp_max_13, convolution_20, squeeze_61, add_107, convolution_21, squeeze_64, clamp_max_14, convolution_22, squeeze_67, clamp_max_15, convolution_23, squeeze_70, add_123, convolution_24, squeeze_73, clamp_max_16, convolution_25, squeeze_76, clamp_max_17, convolution_26, squeeze_79, add_139, convolution_27, squeeze_82, clamp_max_18, convolution_28, squeeze_85, clamp_max_19, convolution_29, squeeze_88, add_155, convolution_30, squeeze_91, clamp_max_20, convolution_31, squeeze_94, clamp_max_21, convolution_32, squeeze_97, add_170, convolution_33, squeeze_100, clamp_max_22, convolution_34, squeeze_103, clamp_max_23, convolution_35, squeeze_106, add_186, convolution_36, squeeze_109, clamp_max_24, convolution_37, squeeze_112, clamp_max_25, convolution_38, squeeze_115, add_202, convolution_39, squeeze_118, clamp_max_26, convolution_40, squeeze_121, clamp_max_27, convolution_41, squeeze_124, add_217, convolution_42, squeeze_127, clamp_max_28, convolution_43, squeeze_130, clamp_max_29, convolution_44, squeeze_133, add_233, convolution_45, squeeze_136, clamp_max_30, convolution_46, squeeze_139, clamp_max_31, convolution_47, squeeze_142, add_249, convolution_48, squeeze_145, clamp_max_32, convolution_49, squeeze_148, clamp_max_33, convolution_50, squeeze_151, add_264, convolution_51, squeeze_154, view, permute_1, bitwise_or, unsqueeze_210, unsqueeze_222, bitwise_or_1, unsqueeze_234, bitwise_or_2, unsqueeze_246, unsqueeze_258, bitwise_or_3, unsqueeze_270, bitwise_or_4, unsqueeze_282, unsqueeze_294, bitwise_or_5, unsqueeze_306, bitwise_or_6, unsqueeze_318, unsqueeze_330, bitwise_or_7, unsqueeze_342, bitwise_or_8, unsqueeze_354, unsqueeze_366, bitwise_or_9, unsqueeze_378, bitwise_or_10, unsqueeze_390, unsqueeze_402, bitwise_or_11, unsqueeze_414, bitwise_or_12, unsqueeze_426, unsqueeze_438, bitwise_or_13, unsqueeze_450, bitwise_or_14, unsqueeze_462, unsqueeze_474, bitwise_or_15, unsqueeze_486, bitwise_or_16, unsqueeze_498, unsqueeze_510, bitwise_or_17, unsqueeze_522, bitwise_or_18, unsqueeze_534, unsqueeze_546, bitwise_or_19, unsqueeze_558, bitwise_or_20, unsqueeze_570, unsqueeze_582, bitwise_or_21, unsqueeze_594, bitwise_or_22, unsqueeze_606, unsqueeze_618, bitwise_or_23, unsqueeze_630, bitwise_or_24, unsqueeze_642, unsqueeze_654, bitwise_or_25, unsqueeze_666, bitwise_or_26, unsqueeze_678, unsqueeze_690, bitwise_or_27, unsqueeze_702, bitwise_or_28, unsqueeze_714, unsqueeze_726, bitwise_or_29, unsqueeze_738, bitwise_or_30, unsqueeze_750, unsqueeze_762, bitwise_or_31, unsqueeze_774, bitwise_or_32, unsqueeze_786, unsqueeze_798, bitwise_or_33, unsqueeze_810, bitwise_or_34, unsqueeze_822, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_13, (144, ), (1, ))
    assert_size_stride(primals_15, (144, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (144, ), (1, ))
    assert_size_stride(primals_21, (144, ), (1, ))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_25, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_65, (96, ), (1, ))
    assert_size_stride(primals_67, (576, ), (1, ))
    assert_size_stride(primals_69, (576, ), (1, ))
    assert_size_stride(primals_71, (96, ), (1, ))
    assert_size_stride(primals_73, (576, ), (1, ))
    assert_size_stride(primals_75, (576, ), (1, ))
    assert_size_stride(primals_77, (96, ), (1, ))
    assert_size_stride(primals_79, (576, ), (1, ))
    assert_size_stride(primals_81, (576, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_85, (960, ), (1, ))
    assert_size_stride(primals_87, (960, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_91, (960, ), (1, ))
    assert_size_stride(primals_93, (960, ), (1, ))
    assert_size_stride(primals_95, (160, ), (1, ))
    assert_size_stride(primals_97, (960, ), (1, ))
    assert_size_stride(primals_99, (960, ), (1, ))
    assert_size_stride(primals_101, (320, ), (1, ))
    assert_size_stride(primals_103, (1280, ), (1, ))
    assert_size_stride(primals_105, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_106, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_108, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_109, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_111, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_112, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_114, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_115, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_116, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_117, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_118, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_120, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_121, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_122, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_123, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_124, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_126, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_127, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_129, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_130, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_132, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_133, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_135, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_136, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_138, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_139, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_140, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_141, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_142, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_144, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_145, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (160, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_147, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_148, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_150, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_151, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_153, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_154, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (320, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_156, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_315, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(clamp_max, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(clamp_max_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_14, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(clamp_max_2, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(convolution_4, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(clamp_max_3, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_5, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_16, (24, ), (1, ))
    assert_size_stride(add_29, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_6, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_19, (144, ), (1, ))
    assert_size_stride(clamp_max_4, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_7, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_22, (144, ), (1, ))
    assert_size_stride(clamp_max_5, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_8, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_25, (24, ), (1, ))
    assert_size_stride(add_45, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_9, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_28, (144, ), (1, ))
    assert_size_stride(clamp_max_6, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_10, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(squeeze_31, (144, ), (1, ))
    assert_size_stride(clamp_max_7, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(convolution_11, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(squeeze_34, (32, ), (1, ))
    assert_size_stride(add_60, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_12, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_37, (192, ), (1, ))
    assert_size_stride(clamp_max_8, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_13, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_40, (192, ), (1, ))
    assert_size_stride(clamp_max_9, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_14, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(squeeze_43, (32, ), (1, ))
    assert_size_stride(add_76, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_15, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_46, (192, ), (1, ))
    assert_size_stride(clamp_max_10, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_16, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_49, (192, ), (1, ))
    assert_size_stride(clamp_max_11, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_17, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(squeeze_52, (32, ), (1, ))
    assert_size_stride(add_92, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_18, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_55, (192, ), (1, ))
    assert_size_stride(clamp_max_12, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_19, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(squeeze_58, (192, ), (1, ))
    assert_size_stride(clamp_max_13, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(convolution_20, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_61, (64, ), (1, ))
    assert_size_stride(add_107, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_21, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_64, (384, ), (1, ))
    assert_size_stride(clamp_max_14, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_22, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_67, (384, ), (1, ))
    assert_size_stride(clamp_max_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_23, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_70, (64, ), (1, ))
    assert_size_stride(add_123, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_24, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_73, (384, ), (1, ))
    assert_size_stride(clamp_max_16, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_25, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_76, (384, ), (1, ))
    assert_size_stride(clamp_max_17, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_26, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_79, (64, ), (1, ))
    assert_size_stride(add_139, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_27, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_82, (384, ), (1, ))
    assert_size_stride(clamp_max_18, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_28, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_85, (384, ), (1, ))
    assert_size_stride(clamp_max_19, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_29, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_88, (64, ), (1, ))
    assert_size_stride(add_155, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_30, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_91, (384, ), (1, ))
    assert_size_stride(clamp_max_20, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_31, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_94, (384, ), (1, ))
    assert_size_stride(clamp_max_21, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_32, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(squeeze_97, (96, ), (1, ))
    assert_size_stride(add_170, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_33, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_100, (576, ), (1, ))
    assert_size_stride(clamp_max_22, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_34, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_103, (576, ), (1, ))
    assert_size_stride(clamp_max_23, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_35, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(squeeze_106, (96, ), (1, ))
    assert_size_stride(add_186, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_36, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_109, (576, ), (1, ))
    assert_size_stride(clamp_max_24, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_37, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_112, (576, ), (1, ))
    assert_size_stride(clamp_max_25, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_38, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(squeeze_115, (96, ), (1, ))
    assert_size_stride(add_202, (8, 96, 14, 14), (18816, 1, 1344, 96))
    assert_size_stride(convolution_39, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(squeeze_118, (576, ), (1, ))
    assert_size_stride(clamp_max_26, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(convolution_40, (8, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(squeeze_121, (576, ), (1, ))
    assert_size_stride(clamp_max_27, (8, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(convolution_41, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_124, (160, ), (1, ))
    assert_size_stride(add_217, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_42, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_127, (960, ), (1, ))
    assert_size_stride(clamp_max_28, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_43, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_130, (960, ), (1, ))
    assert_size_stride(clamp_max_29, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_44, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_133, (160, ), (1, ))
    assert_size_stride(add_233, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_45, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_136, (960, ), (1, ))
    assert_size_stride(clamp_max_30, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_46, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_139, (960, ), (1, ))
    assert_size_stride(clamp_max_31, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_47, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_142, (160, ), (1, ))
    assert_size_stride(add_249, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_48, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_145, (960, ), (1, ))
    assert_size_stride(clamp_max_32, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_49, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_148, (960, ), (1, ))
    assert_size_stride(clamp_max_33, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_50, (8, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(squeeze_151, (320, ), (1, ))
    assert_size_stride(add_264, (8, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(convolution_51, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(squeeze_154, (1280, ), (1, ))
    assert_size_stride(view, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(bitwise_or, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(unsqueeze_210, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_222, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(bitwise_or_1, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_234, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(bitwise_or_2, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_246, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_or_3, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_270, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(bitwise_or_4, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_282, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_or_5, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_306, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(bitwise_or_6, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_318, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(bitwise_or_7, (8, 576, 7, 7), (28224, 1, 4032, 576))
    assert_size_stride(unsqueeze_342, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(bitwise_or_8, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_354, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(bitwise_or_9, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_378, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(bitwise_or_10, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_390, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(bitwise_or_11, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_414, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(bitwise_or_12, (8, 576, 14, 14), (112896, 1, 8064, 576))
    assert_size_stride(unsqueeze_426, (1, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(bitwise_or_13, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_450, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(bitwise_or_14, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_462, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(bitwise_or_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_486, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(bitwise_or_16, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_498, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(bitwise_or_17, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_522, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(bitwise_or_18, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_534, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(bitwise_or_19, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_558, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(bitwise_or_20, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(unsqueeze_570, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(bitwise_or_21, (8, 192, 14, 14), (37632, 1, 2688, 192))
    assert_size_stride(unsqueeze_594, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(bitwise_or_22, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_606, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(bitwise_or_23, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_630, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(bitwise_or_24, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_642, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(bitwise_or_25, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_666, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(bitwise_or_26, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(unsqueeze_678, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(bitwise_or_27, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(unsqueeze_702, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(bitwise_or_28, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_714, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(bitwise_or_29, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_738, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(bitwise_or_30, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_750, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(bitwise_or_31, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(unsqueeze_774, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(bitwise_or_32, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(unsqueeze_786, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_798, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(bitwise_or_33, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_810, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(bitwise_or_34, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_822, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view, out=buf1)
    del view
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf4 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf5 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_hardtanh_backward_native_batch_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(bitwise_or.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_210.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del bitwise_or
    del buf0
    del buf4
    del convolution_51
    del primals_103
    del squeeze_154
    del tangents_1
    del unsqueeze_210
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf7 = aten.convolution_backward(buf6, add_264, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_264
    del buf6
    del primals_156
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((320, ), device='cpu', dtype=torch.float32)
    buf11 = empty((320, ), device='cpu', dtype=torch.float32)
    buf12 = empty((320, ), device='cpu', dtype=torch.float32)
    buf13 = buf8; del buf8  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_1(c_void_p(buf13.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_222.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del buf11
    del convolution_50
    del primals_101
    del squeeze_151
    del unsqueeze_222
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf14 = aten.convolution_backward(buf13, clamp_max_33, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf13
    del clamp_max_33
    del primals_155
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty((960, ), device='cpu', dtype=torch.float32)
    buf18 = empty((960, ), device='cpu', dtype=torch.float32)
    buf19 = empty((960, ), device='cpu', dtype=torch.float32)
    buf20 = buf15; del buf15  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_2(c_void_p(buf20.data_ptr()), c_void_p(bitwise_or_1.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    del bitwise_or_1
    del convolution_49
    del primals_99
    del squeeze_148
    del unsqueeze_234
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf21 = aten.convolution_backward(buf20, clamp_max_32, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False])
    del buf20
    del clamp_max_32
    del primals_154
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    buf24 = buf18; del buf18  # reuse
    buf25 = empty((960, ), device='cpu', dtype=torch.float32)
    buf26 = empty((960, ), device='cpu', dtype=torch.float32)
    buf27 = buf22; del buf22  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_3(c_void_p(buf27.data_ptr()), c_void_p(bitwise_or_2.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del bitwise_or_2
    del convolution_48
    del primals_97
    del squeeze_145
    del unsqueeze_246
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf28 = aten.convolution_backward(buf27, add_249, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_249
    del buf27
    del primals_153
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf31 = empty((160, ), device='cpu', dtype=torch.float32)
    buf32 = empty((160, ), device='cpu', dtype=torch.float32)
    buf33 = empty((160, ), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_4(c_void_p(buf29.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del convolution_47
    del primals_95
    del squeeze_142
    del unsqueeze_258
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf35 = aten.convolution_backward(buf34, clamp_max_31, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clamp_max_31
    del primals_152
    buf36 = buf35[0]
    buf37 = buf35[1]
    del buf35
    buf38 = buf25; del buf25  # reuse
    buf39 = empty((960, ), device='cpu', dtype=torch.float32)
    buf40 = empty((960, ), device='cpu', dtype=torch.float32)
    buf41 = buf36; del buf36  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_5(c_void_p(buf41.data_ptr()), c_void_p(bitwise_or_3.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_270.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    del bitwise_or_3
    del convolution_46
    del primals_93
    del squeeze_139
    del unsqueeze_270
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf42 = aten.convolution_backward(buf41, clamp_max_30, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False])
    del buf41
    del clamp_max_30
    del primals_151
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    buf45 = buf39; del buf39  # reuse
    buf46 = empty((960, ), device='cpu', dtype=torch.float32)
    buf47 = empty((960, ), device='cpu', dtype=torch.float32)
    buf48 = buf43; del buf43  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_6(c_void_p(buf48.data_ptr()), c_void_p(bitwise_or_4.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    del bitwise_or_4
    del convolution_45
    del primals_91
    del squeeze_136
    del unsqueeze_282
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf49 = aten.convolution_backward(buf48, add_233, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_233
    del buf48
    del primals_150
    buf50 = buf49[0]
    buf51 = buf49[1]
    del buf49
    buf52 = buf32; del buf32  # reuse
    buf53 = empty((160, ), device='cpu', dtype=torch.float32)
    buf54 = empty((160, ), device='cpu', dtype=torch.float32)
    buf55 = buf34; del buf34  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_7(c_void_p(buf29.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_294.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del convolution_44
    del primals_89
    del squeeze_133
    del unsqueeze_294
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf56 = aten.convolution_backward(buf55, clamp_max_29, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf55
    del clamp_max_29
    del primals_149
    buf57 = buf56[0]
    buf58 = buf56[1]
    del buf56
    buf59 = buf46; del buf46  # reuse
    buf60 = empty((960, ), device='cpu', dtype=torch.float32)
    buf61 = empty((960, ), device='cpu', dtype=torch.float32)
    buf62 = buf57; del buf57  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_8(c_void_p(buf62.data_ptr()), c_void_p(bitwise_or_5.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    del bitwise_or_5
    del convolution_43
    del primals_87
    del squeeze_130
    del unsqueeze_306
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf63 = aten.convolution_backward(buf62, clamp_max_28, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False])
    del buf62
    del clamp_max_28
    del primals_148
    buf64 = buf63[0]
    buf65 = buf63[1]
    del buf63
    buf66 = buf60; del buf60  # reuse
    buf67 = empty((960, ), device='cpu', dtype=torch.float32)
    buf68 = empty((960, ), device='cpu', dtype=torch.float32)
    buf69 = buf64; del buf64  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_9(c_void_p(buf69.data_ptr()), c_void_p(bitwise_or_6.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_318.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del bitwise_or_6
    del buf67
    del convolution_42
    del primals_85
    del squeeze_127
    del unsqueeze_318
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf70 = aten.convolution_backward(buf69, add_217, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_217
    del buf69
    del primals_147
    buf71 = buf70[0]
    buf72 = buf70[1]
    del buf70
    buf73 = buf53; del buf53  # reuse
    buf74 = empty((160, ), device='cpu', dtype=torch.float32)
    buf75 = buf29; del buf29  # reuse
    buf76 = buf74; del buf74  # reuse
    cpp_fused_add_native_batch_norm_backward_10(c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf73.data_ptr()))
    del buf50
    del buf71
    del convolution_41
    del primals_83
    del squeeze_124
    del unsqueeze_330
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf77 = aten.convolution_backward(buf75, clamp_max_27, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf75
    del clamp_max_27
    del primals_146
    buf78 = buf77[0]
    buf79 = buf77[1]
    del buf77
    buf80 = empty((576, ), device='cpu', dtype=torch.float32)
    buf81 = empty((576, ), device='cpu', dtype=torch.float32)
    buf82 = empty((576, ), device='cpu', dtype=torch.float32)
    buf83 = buf78; del buf78  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_11(c_void_p(buf83.data_ptr()), c_void_p(bitwise_or_7.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    del bitwise_or_7
    del convolution_40
    del primals_81
    del squeeze_121
    del unsqueeze_342
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf84 = aten.convolution_backward(buf83, clamp_max_26, primals_145, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
    del buf83
    del clamp_max_26
    del primals_145
    buf85 = buf84[0]
    buf86 = buf84[1]
    del buf84
    buf87 = buf81; del buf81  # reuse
    buf88 = empty((576, ), device='cpu', dtype=torch.float32)
    buf89 = empty((576, ), device='cpu', dtype=torch.float32)
    buf90 = buf85; del buf85  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_12(c_void_p(buf90.data_ptr()), c_void_p(bitwise_or_8.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    del bitwise_or_8
    del convolution_39
    del primals_79
    del squeeze_118
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf91 = aten.convolution_backward(buf90, add_202, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_202
    del buf90
    del primals_144
    buf92 = buf91[0]
    buf93 = buf91[1]
    del buf91
    buf94 = empty((96, ), device='cpu', dtype=torch.float32)
    buf95 = empty((96, ), device='cpu', dtype=torch.float32)
    buf96 = empty((96, ), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_13(c_void_p(buf92.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del convolution_38
    del primals_77
    del squeeze_115
    del unsqueeze_366
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf98 = aten.convolution_backward(buf97, clamp_max_25, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clamp_max_25
    del primals_143
    buf99 = buf98[0]
    buf100 = buf98[1]
    del buf98
    buf101 = buf88; del buf88  # reuse
    buf102 = empty((576, ), device='cpu', dtype=torch.float32)
    buf103 = empty((576, ), device='cpu', dtype=torch.float32)
    buf104 = buf99; del buf99  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_14(c_void_p(buf104.data_ptr()), c_void_p(bitwise_or_9.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    del bitwise_or_9
    del convolution_37
    del primals_75
    del squeeze_112
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf105 = aten.convolution_backward(buf104, clamp_max_24, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
    del buf104
    del clamp_max_24
    del primals_142
    buf106 = buf105[0]
    buf107 = buf105[1]
    del buf105
    buf108 = buf102; del buf102  # reuse
    buf109 = empty((576, ), device='cpu', dtype=torch.float32)
    buf110 = empty((576, ), device='cpu', dtype=torch.float32)
    buf111 = buf106; del buf106  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_15(c_void_p(buf111.data_ptr()), c_void_p(bitwise_or_10.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    del bitwise_or_10
    del convolution_36
    del primals_73
    del squeeze_109
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf112 = aten.convolution_backward(buf111, add_186, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_186
    del buf111
    del primals_141
    buf113 = buf112[0]
    buf114 = buf112[1]
    del buf112
    buf115 = buf95; del buf95  # reuse
    buf116 = empty((96, ), device='cpu', dtype=torch.float32)
    buf117 = empty((96, ), device='cpu', dtype=torch.float32)
    buf118 = buf97; del buf97  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_16(c_void_p(buf92.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()))
    del convolution_35
    del primals_71
    del squeeze_106
    del unsqueeze_402
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf119 = aten.convolution_backward(buf118, clamp_max_23, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf118
    del clamp_max_23
    del primals_140
    buf120 = buf119[0]
    buf121 = buf119[1]
    del buf119
    buf122 = buf109; del buf109  # reuse
    buf123 = empty((576, ), device='cpu', dtype=torch.float32)
    buf124 = empty((576, ), device='cpu', dtype=torch.float32)
    buf125 = buf120; del buf120  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_17(c_void_p(buf125.data_ptr()), c_void_p(bitwise_or_11.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()))
    del bitwise_or_11
    del convolution_34
    del primals_69
    del squeeze_103
    del unsqueeze_414
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf126 = aten.convolution_backward(buf125, clamp_max_22, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False])
    del buf125
    del clamp_max_22
    del primals_139
    buf127 = buf126[0]
    buf128 = buf126[1]
    del buf126
    buf129 = buf123; del buf123  # reuse
    buf130 = empty((576, ), device='cpu', dtype=torch.float32)
    buf131 = empty((576, ), device='cpu', dtype=torch.float32)
    buf132 = buf127; del buf127  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_18(c_void_p(buf132.data_ptr()), c_void_p(bitwise_or_12.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del bitwise_or_12
    del buf130
    del convolution_33
    del primals_67
    del squeeze_100
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf133 = aten.convolution_backward(buf132, add_170, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_170
    del buf132
    del primals_138
    buf134 = buf133[0]
    buf135 = buf133[1]
    del buf133
    buf136 = buf116; del buf116  # reuse
    buf137 = empty((96, ), device='cpu', dtype=torch.float32)
    buf138 = buf113; del buf113  # reuse
    buf139 = buf137; del buf137  # reuse
    cpp_fused_add_native_batch_norm_backward_19(c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf136.data_ptr()))
    del buf134
    del buf92
    del convolution_32
    del primals_65
    del squeeze_97
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf140 = aten.convolution_backward(buf138, clamp_max_21, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf138
    del clamp_max_21
    del primals_137
    buf141 = buf140[0]
    buf142 = buf140[1]
    del buf140
    buf143 = empty((384, ), device='cpu', dtype=torch.float32)
    buf144 = empty((384, ), device='cpu', dtype=torch.float32)
    buf145 = empty((384, ), device='cpu', dtype=torch.float32)
    buf146 = buf141; del buf141  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_20(c_void_p(buf146.data_ptr()), c_void_p(bitwise_or_13.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del bitwise_or_13
    del convolution_31
    del primals_63
    del squeeze_94
    del unsqueeze_450
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf147 = aten.convolution_backward(buf146, clamp_max_20, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
    del buf146
    del clamp_max_20
    del primals_136
    buf148 = buf147[0]
    buf149 = buf147[1]
    del buf147
    buf150 = buf144; del buf144  # reuse
    buf151 = empty((384, ), device='cpu', dtype=torch.float32)
    buf152 = empty((384, ), device='cpu', dtype=torch.float32)
    buf153 = buf148; del buf148  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_21(c_void_p(buf153.data_ptr()), c_void_p(bitwise_or_14.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    del bitwise_or_14
    del convolution_30
    del primals_61
    del squeeze_91
    del unsqueeze_462
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf154 = aten.convolution_backward(buf153, add_155, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_155
    del buf153
    del primals_135
    buf155 = buf154[0]
    buf156 = buf154[1]
    del buf154
    buf157 = empty((64, ), device='cpu', dtype=torch.float32)
    buf158 = empty((64, ), device='cpu', dtype=torch.float32)
    buf159 = empty((64, ), device='cpu', dtype=torch.float32)
    buf160 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_22(c_void_p(buf155.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del convolution_29
    del primals_59
    del squeeze_88
    del unsqueeze_474
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf161 = aten.convolution_backward(buf160, clamp_max_19, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clamp_max_19
    del primals_134
    buf162 = buf161[0]
    buf163 = buf161[1]
    del buf161
    buf164 = buf151; del buf151  # reuse
    buf165 = empty((384, ), device='cpu', dtype=torch.float32)
    buf166 = empty((384, ), device='cpu', dtype=torch.float32)
    buf167 = buf162; del buf162  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_23(c_void_p(buf167.data_ptr()), c_void_p(bitwise_or_15.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    del bitwise_or_15
    del convolution_28
    del primals_57
    del squeeze_85
    del unsqueeze_486
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf168 = aten.convolution_backward(buf167, clamp_max_18, primals_133, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
    del buf167
    del clamp_max_18
    del primals_133
    buf169 = buf168[0]
    buf170 = buf168[1]
    del buf168
    buf171 = buf165; del buf165  # reuse
    buf172 = empty((384, ), device='cpu', dtype=torch.float32)
    buf173 = empty((384, ), device='cpu', dtype=torch.float32)
    buf174 = buf169; del buf169  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_24(c_void_p(buf174.data_ptr()), c_void_p(bitwise_or_16.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del bitwise_or_16
    del convolution_27
    del primals_55
    del squeeze_82
    del unsqueeze_498
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf175 = aten.convolution_backward(buf174, add_139, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_139
    del buf174
    del primals_132
    buf176 = buf175[0]
    buf177 = buf175[1]
    del buf175
    buf178 = buf158; del buf158  # reuse
    buf179 = empty((64, ), device='cpu', dtype=torch.float32)
    buf180 = empty((64, ), device='cpu', dtype=torch.float32)
    buf181 = buf160; del buf160  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_25(c_void_p(buf155.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del convolution_26
    del primals_53
    del squeeze_79
    del unsqueeze_510
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf182 = aten.convolution_backward(buf181, clamp_max_17, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clamp_max_17
    del primals_131
    buf183 = buf182[0]
    buf184 = buf182[1]
    del buf182
    buf185 = buf172; del buf172  # reuse
    buf186 = empty((384, ), device='cpu', dtype=torch.float32)
    buf187 = empty((384, ), device='cpu', dtype=torch.float32)
    buf188 = buf183; del buf183  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_26(c_void_p(buf188.data_ptr()), c_void_p(bitwise_or_17.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_522.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    del bitwise_or_17
    del convolution_25
    del primals_51
    del squeeze_76
    del unsqueeze_522
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf189 = aten.convolution_backward(buf188, clamp_max_16, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
    del buf188
    del clamp_max_16
    del primals_130
    buf190 = buf189[0]
    buf191 = buf189[1]
    del buf189
    buf192 = buf186; del buf186  # reuse
    buf193 = empty((384, ), device='cpu', dtype=torch.float32)
    buf194 = empty((384, ), device='cpu', dtype=torch.float32)
    buf195 = buf190; del buf190  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_27(c_void_p(buf195.data_ptr()), c_void_p(bitwise_or_18.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_534.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    del bitwise_or_18
    del convolution_24
    del primals_49
    del squeeze_73
    del unsqueeze_534
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf196 = aten.convolution_backward(buf195, add_123, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_123
    del buf195
    del primals_129
    buf197 = buf196[0]
    buf198 = buf196[1]
    del buf196
    buf199 = buf179; del buf179  # reuse
    buf200 = empty((64, ), device='cpu', dtype=torch.float32)
    buf201 = buf181; del buf181  # reuse
    buf202 = buf200; del buf200  # reuse
    cpp_fused_add_native_batch_norm_backward_28(c_void_p(buf202.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf201.data_ptr()))
    del convolution_23
    del primals_47
    del squeeze_70
    del unsqueeze_546
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf203 = aten.convolution_backward(buf201, clamp_max_15, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf201
    del clamp_max_15
    del primals_128
    buf204 = buf203[0]
    buf205 = buf203[1]
    del buf203
    buf206 = buf193; del buf193  # reuse
    buf207 = empty((384, ), device='cpu', dtype=torch.float32)
    buf208 = empty((384, ), device='cpu', dtype=torch.float32)
    buf209 = buf204; del buf204  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_29(c_void_p(buf209.data_ptr()), c_void_p(bitwise_or_19.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_558.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del bitwise_or_19
    del convolution_22
    del primals_45
    del squeeze_67
    del unsqueeze_558
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf210 = aten.convolution_backward(buf209, clamp_max_14, primals_127, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False])
    del buf209
    del clamp_max_14
    del primals_127
    buf211 = buf210[0]
    buf212 = buf210[1]
    del buf210
    buf213 = buf207; del buf207  # reuse
    buf214 = empty((384, ), device='cpu', dtype=torch.float32)
    buf215 = empty((384, ), device='cpu', dtype=torch.float32)
    buf216 = buf211; del buf211  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_30(c_void_p(buf216.data_ptr()), c_void_p(bitwise_or_20.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_570.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    del bitwise_or_20
    del buf214
    del convolution_21
    del primals_43
    del squeeze_64
    del unsqueeze_570
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf217 = aten.convolution_backward(buf216, add_107, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_107
    del primals_126
    buf218 = buf217[0]
    buf219 = buf217[1]
    del buf217
    buf220 = empty((64, ), device='cpu', dtype=torch.float32)
    buf221 = empty((64, ), device='cpu', dtype=torch.float32)
    buf222 = buf155; del buf155  # reuse
    buf224 = buf222; del buf222  # reuse
    buf223 = buf221; del buf221  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_31(c_void_p(buf224.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf220.data_ptr()))
    del buf176
    del buf197
    del buf218
    del convolution_20
    del primals_41
    del squeeze_61
    del unsqueeze_582
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf225 = aten.convolution_backward(buf224, clamp_max_13, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf224
    del clamp_max_13
    del primals_125
    buf226 = buf225[0]
    buf227 = buf225[1]
    del buf225
    buf228 = empty((192, ), device='cpu', dtype=torch.float32)
    buf229 = empty((192, ), device='cpu', dtype=torch.float32)
    buf230 = empty((192, ), device='cpu', dtype=torch.float32)
    buf231 = buf226; del buf226  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_32(c_void_p(buf231.data_ptr()), c_void_p(bitwise_or_21.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_594.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    del bitwise_or_21
    del convolution_19
    del primals_39
    del squeeze_58
    del unsqueeze_594
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf232 = aten.convolution_backward(buf231, clamp_max_12, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
    del buf231
    del clamp_max_12
    del primals_124
    buf233 = buf232[0]
    buf234 = buf232[1]
    del buf232
    buf235 = buf229; del buf229  # reuse
    buf236 = empty((192, ), device='cpu', dtype=torch.float32)
    buf237 = empty((192, ), device='cpu', dtype=torch.float32)
    buf238 = buf233; del buf233  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_33(c_void_p(buf238.data_ptr()), c_void_p(bitwise_or_22.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_606.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    del bitwise_or_22
    del convolution_18
    del primals_37
    del squeeze_55
    del unsqueeze_606
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf239 = aten.convolution_backward(buf238, add_92, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_92
    del buf238
    del primals_123
    buf240 = buf239[0]
    buf241 = buf239[1]
    del buf239
    buf242 = empty((32, ), device='cpu', dtype=torch.float32)
    buf243 = empty((32, ), device='cpu', dtype=torch.float32)
    buf244 = empty((32, ), device='cpu', dtype=torch.float32)
    buf245 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_34(c_void_p(buf240.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    del convolution_17
    del primals_35
    del squeeze_52
    del unsqueeze_618
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf246 = aten.convolution_backward(buf245, clamp_max_11, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clamp_max_11
    del primals_122
    buf247 = buf246[0]
    buf248 = buf246[1]
    del buf246
    buf249 = buf236; del buf236  # reuse
    buf250 = empty((192, ), device='cpu', dtype=torch.float32)
    buf251 = empty((192, ), device='cpu', dtype=torch.float32)
    buf252 = buf247; del buf247  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_35(c_void_p(buf252.data_ptr()), c_void_p(bitwise_or_23.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_630.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    del bitwise_or_23
    del convolution_16
    del primals_33
    del squeeze_49
    del unsqueeze_630
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf253 = aten.convolution_backward(buf252, clamp_max_10, primals_121, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
    del buf252
    del clamp_max_10
    del primals_121
    buf254 = buf253[0]
    buf255 = buf253[1]
    del buf253
    buf256 = buf250; del buf250  # reuse
    buf257 = empty((192, ), device='cpu', dtype=torch.float32)
    buf258 = empty((192, ), device='cpu', dtype=torch.float32)
    buf259 = buf254; del buf254  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_36(c_void_p(buf259.data_ptr()), c_void_p(bitwise_or_24.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_642.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del bitwise_or_24
    del convolution_15
    del primals_31
    del squeeze_46
    del unsqueeze_642
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf260 = aten.convolution_backward(buf259, add_76, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_76
    del buf259
    del primals_120
    buf261 = buf260[0]
    buf262 = buf260[1]
    del buf260
    buf263 = buf243; del buf243  # reuse
    buf264 = empty((32, ), device='cpu', dtype=torch.float32)
    buf265 = empty((32, ), device='cpu', dtype=torch.float32)
    buf266 = buf245; del buf245  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_37(c_void_p(buf240.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    del convolution_14
    del primals_29
    del squeeze_43
    del unsqueeze_654
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf267 = aten.convolution_backward(buf266, clamp_max_9, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf266
    del clamp_max_9
    del primals_119
    buf268 = buf267[0]
    buf269 = buf267[1]
    del buf267
    buf270 = buf257; del buf257  # reuse
    buf271 = empty((192, ), device='cpu', dtype=torch.float32)
    buf272 = empty((192, ), device='cpu', dtype=torch.float32)
    buf273 = buf268; del buf268  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_38(c_void_p(buf273.data_ptr()), c_void_p(bitwise_or_25.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_666.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    del bitwise_or_25
    del convolution_13
    del primals_27
    del squeeze_40
    del unsqueeze_666
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf274 = aten.convolution_backward(buf273, clamp_max_8, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False])
    del buf273
    del clamp_max_8
    del primals_118
    buf275 = buf274[0]
    buf276 = buf274[1]
    del buf274
    buf277 = buf271; del buf271  # reuse
    buf278 = empty((192, ), device='cpu', dtype=torch.float32)
    buf279 = empty((192, ), device='cpu', dtype=torch.float32)
    buf280 = buf275; del buf275  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_39(c_void_p(buf280.data_ptr()), c_void_p(bitwise_or_26.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_678.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    del bitwise_or_26
    del buf278
    del convolution_12
    del primals_25
    del squeeze_37
    del unsqueeze_678
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf281 = aten.convolution_backward(buf280, add_60, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_60
    del buf280
    del primals_117
    buf282 = buf281[0]
    buf283 = buf281[1]
    del buf281
    buf284 = buf264; del buf264  # reuse
    buf285 = empty((32, ), device='cpu', dtype=torch.float32)
    buf286 = buf240; del buf240  # reuse
    buf287 = buf285; del buf285  # reuse
    cpp_fused_add_native_batch_norm_backward_40(c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf284.data_ptr()))
    del buf261
    del buf282
    del convolution_11
    del primals_23
    del squeeze_34
    del unsqueeze_690
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf288 = aten.convolution_backward(buf286, clamp_max_7, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf286
    del clamp_max_7
    del primals_116
    buf289 = buf288[0]
    buf290 = buf288[1]
    del buf288
    buf291 = empty((144, ), device='cpu', dtype=torch.float32)
    buf292 = empty((144, ), device='cpu', dtype=torch.float32)
    buf293 = empty((144, ), device='cpu', dtype=torch.float32)
    buf294 = buf289; del buf289  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_41(c_void_p(buf294.data_ptr()), c_void_p(bitwise_or_27.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_702.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    del bitwise_or_27
    del convolution_10
    del primals_21
    del squeeze_31
    del unsqueeze_702
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf295 = aten.convolution_backward(buf294, clamp_max_6, primals_115, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
    del buf294
    del clamp_max_6
    del primals_115
    buf296 = buf295[0]
    buf297 = buf295[1]
    del buf295
    buf298 = buf292; del buf292  # reuse
    buf299 = empty((144, ), device='cpu', dtype=torch.float32)
    buf300 = empty((144, ), device='cpu', dtype=torch.float32)
    buf301 = buf296; del buf296  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_42(c_void_p(buf301.data_ptr()), c_void_p(bitwise_or_28.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_714.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    del bitwise_or_28
    del convolution_9
    del primals_19
    del squeeze_28
    del unsqueeze_714
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf302 = aten.convolution_backward(buf301, add_45, primals_114, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_45
    del buf301
    del primals_114
    buf303 = buf302[0]
    buf304 = buf302[1]
    del buf302
    buf305 = empty((24, ), device='cpu', dtype=torch.float32)
    buf306 = empty((24, ), device='cpu', dtype=torch.float32)
    buf307 = empty((24, ), device='cpu', dtype=torch.float32)
    buf308 = reinterpret_tensor(buf216, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf216  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_43(c_void_p(buf303.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    del convolution_8
    del primals_17
    del squeeze_25
    del unsqueeze_726
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf309 = aten.convolution_backward(buf308, clamp_max_5, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf308
    del clamp_max_5
    del primals_113
    buf310 = buf309[0]
    buf311 = buf309[1]
    del buf309
    buf312 = buf299; del buf299  # reuse
    buf313 = empty((144, ), device='cpu', dtype=torch.float32)
    buf314 = empty((144, ), device='cpu', dtype=torch.float32)
    buf315 = buf310; del buf310  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_44(c_void_p(buf315.data_ptr()), c_void_p(bitwise_or_29.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_738.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    del bitwise_or_29
    del convolution_7
    del primals_15
    del squeeze_22
    del unsqueeze_738
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf316 = aten.convolution_backward(buf315, clamp_max_4, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
    del buf315
    del clamp_max_4
    del primals_112
    buf317 = buf316[0]
    buf318 = buf316[1]
    del buf316
    buf319 = buf313; del buf313  # reuse
    buf320 = empty((144, ), device='cpu', dtype=torch.float32)
    buf321 = empty((144, ), device='cpu', dtype=torch.float32)
    buf322 = buf317; del buf317  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_45(c_void_p(buf322.data_ptr()), c_void_p(bitwise_or_30.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_750.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    del bitwise_or_30
    del buf320
    del convolution_6
    del primals_13
    del squeeze_19
    del unsqueeze_750
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf323 = aten.convolution_backward(buf322, add_29, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_29
    del buf322
    del primals_111
    buf324 = buf323[0]
    buf325 = buf323[1]
    del buf323
    buf326 = buf306; del buf306  # reuse
    buf327 = empty((24, ), device='cpu', dtype=torch.float32)
    buf328 = empty((24, ), device='cpu', dtype=torch.float32)
    buf329 = buf303; del buf303  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_46(c_void_p(buf329.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_762.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    del buf324
    del buf327
    del convolution_5
    del primals_11
    del squeeze_16
    del unsqueeze_762
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf330 = aten.convolution_backward(buf329, clamp_max_3, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf329
    del clamp_max_3
    del primals_110
    buf331 = buf330[0]
    buf332 = buf330[1]
    del buf330
    buf333 = empty((96, ), device='cpu', dtype=torch.float32)
    buf334 = empty((96, ), device='cpu', dtype=torch.float32)
    buf335 = empty((96, ), device='cpu', dtype=torch.float32)
    buf336 = buf331; del buf331  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_47(c_void_p(buf336.data_ptr()), c_void_p(bitwise_or_31.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_774.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()))
    del bitwise_or_31
    del convolution_4
    del primals_9
    del squeeze_13
    del unsqueeze_774
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf337 = aten.convolution_backward(buf336, clamp_max_2, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False])
    del buf336
    del clamp_max_2
    del primals_109
    buf338 = buf337[0]
    buf339 = buf337[1]
    del buf337
    buf340 = buf334; del buf334  # reuse
    buf341 = empty((96, ), device='cpu', dtype=torch.float32)
    buf342 = empty((96, ), device='cpu', dtype=torch.float32)
    buf343 = buf338; del buf338  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_48(c_void_p(buf343.data_ptr()), c_void_p(bitwise_or_32.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_786.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    del bitwise_or_32
    del buf341
    del convolution_3
    del primals_7
    del squeeze_10
    del unsqueeze_786
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf344 = aten.convolution_backward(buf343, add_14, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_14
    del buf343
    del primals_108
    buf345 = buf344[0]
    buf346 = buf344[1]
    del buf344
    buf347 = empty((16, ), device='cpu', dtype=torch.float32)
    buf348 = empty((16, ), device='cpu', dtype=torch.float32)
    buf349 = empty((16, ), device='cpu', dtype=torch.float32)
    buf350 = buf345; del buf345  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_49(c_void_p(buf350.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_798.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()))
    del buf348
    del convolution_2
    del primals_5
    del squeeze_7
    del unsqueeze_798
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf351 = aten.convolution_backward(buf350, clamp_max_1, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf350
    del clamp_max_1
    del primals_107
    buf352 = buf351[0]
    buf353 = buf351[1]
    del buf351
    buf354 = empty((32, ), device='cpu', dtype=torch.float32)
    buf355 = empty((32, ), device='cpu', dtype=torch.float32)
    buf356 = empty((32, ), device='cpu', dtype=torch.float32)
    buf357 = buf352; del buf352  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_50(c_void_p(buf357.data_ptr()), c_void_p(bitwise_or_33.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_810.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()))
    del bitwise_or_33
    del convolution_1
    del primals_3
    del squeeze_4
    del unsqueeze_810
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf358 = aten.convolution_backward(buf357, clamp_max, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf357
    del clamp_max
    del primals_106
    buf359 = buf358[0]
    buf360 = buf358[1]
    del buf358
    buf361 = buf355; del buf355  # reuse
    buf362 = empty((32, ), device='cpu', dtype=torch.float32)
    buf363 = empty((32, ), device='cpu', dtype=torch.float32)
    buf364 = buf359; del buf359  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_51(c_void_p(buf364.data_ptr()), c_void_p(bitwise_or_34.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_822.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    del bitwise_or_34
    del buf362
    del convolution
    del primals_1
    del squeeze_1
    del unsqueeze_822
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf365 = aten.convolution_backward(buf364, primals_315, primals_105, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf364
    del primals_105
    del primals_315
    buf366 = buf365[1]
    return (buf363, buf361, buf356, buf354, buf349, buf347, buf342, buf340, buf335, buf333, buf328, buf326, buf321, buf319, buf314, buf312, buf307, buf305, buf300, buf298, buf293, buf291, buf287, buf284, buf279, buf277, buf272, buf270, buf265, buf263, buf258, buf256, buf251, buf249, buf244, buf242, buf237, buf235, buf230, buf228, buf223, buf220, buf215, buf213, buf208, buf206, buf202, buf199, buf194, buf192, buf187, buf185, buf180, buf178, buf173, buf171, buf166, buf164, buf159, buf157, buf152, buf150, buf145, buf143, buf139, buf136, buf131, buf129, buf124, buf122, buf117, buf115, buf110, buf108, buf103, buf101, buf96, buf94, buf89, buf87, buf82, buf80, buf76, buf73, buf68, buf66, buf61, buf59, buf54, buf52, buf47, buf45, buf40, buf38, buf33, buf31, buf26, buf24, buf19, buf17, buf12, buf10, buf5, buf3, buf366, buf360, buf353, buf346, buf339, buf332, buf325, buf318, buf311, buf304, buf297, buf290, buf283, buf276, buf269, buf262, buf255, buf248, buf241, buf234, buf227, buf219, buf212, buf205, buf198, buf191, buf184, buf177, buf170, buf163, buf156, buf149, buf142, buf135, buf128, buf121, buf114, buf107, buf100, buf93, buf86, buf79, buf72, buf65, buf58, buf51, buf44, buf37, buf30, buf23, buf16, buf9, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((160, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((320, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    add_14 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_2 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_3 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_29 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_4 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_5 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_45 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_6 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_7 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_60 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_8 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_9 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_76 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_10 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_11 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_92 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_12 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_13 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_107 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_14 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_123 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_16 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_17 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_139 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_18 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_19 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_155 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_20 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_21 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    add_170 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_22 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_23 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    add_186 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_24 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_25 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    add_202 = rand_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_26 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_27 = rand_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    add_217 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_28 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_29 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    add_233 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_30 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_31 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    add_249 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_32 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_33 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    add_264 = rand_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    view = rand_strided((8, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    bitwise_or = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.bool)
    unsqueeze_210 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_1 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.bool)
    unsqueeze_234 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_2 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.bool)
    unsqueeze_246 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_3 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.bool)
    unsqueeze_270 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_4 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.bool)
    unsqueeze_282 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_5 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.bool)
    unsqueeze_306 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_6 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.bool)
    unsqueeze_318 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_7 = rand_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cpu', dtype=torch.bool)
    unsqueeze_342 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_8 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.bool)
    unsqueeze_354 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_9 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.bool)
    unsqueeze_378 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_10 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.bool)
    unsqueeze_390 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_11 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.bool)
    unsqueeze_414 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_12 = rand_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cpu', dtype=torch.bool)
    unsqueeze_426 = rand_strided((1, 576, 1, 1), (576, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_13 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.bool)
    unsqueeze_450 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_14 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.bool)
    unsqueeze_462 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.bool)
    unsqueeze_486 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_16 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.bool)
    unsqueeze_498 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_17 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.bool)
    unsqueeze_522 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_18 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.bool)
    unsqueeze_534 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_19 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_20 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.bool)
    unsqueeze_570 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_21 = rand_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cpu', dtype=torch.bool)
    unsqueeze_594 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_22 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.bool)
    unsqueeze_606 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_23 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.bool)
    unsqueeze_630 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_24 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.bool)
    unsqueeze_642 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_25 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.bool)
    unsqueeze_666 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_26 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.bool)
    unsqueeze_678 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_27 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cpu', dtype=torch.bool)
    unsqueeze_702 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_28 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.bool)
    unsqueeze_714 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_29 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.bool)
    unsqueeze_738 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_30 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.bool)
    unsqueeze_750 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_31 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.bool)
    unsqueeze_774 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_32 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.bool)
    unsqueeze_786 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_798 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_33 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.bool)
    unsqueeze_810 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_34 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.bool)
    unsqueeze_822 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_315, convolution, squeeze_1, clamp_max, convolution_1, squeeze_4, clamp_max_1, convolution_2, squeeze_7, add_14, convolution_3, squeeze_10, clamp_max_2, convolution_4, squeeze_13, clamp_max_3, convolution_5, squeeze_16, add_29, convolution_6, squeeze_19, clamp_max_4, convolution_7, squeeze_22, clamp_max_5, convolution_8, squeeze_25, add_45, convolution_9, squeeze_28, clamp_max_6, convolution_10, squeeze_31, clamp_max_7, convolution_11, squeeze_34, add_60, convolution_12, squeeze_37, clamp_max_8, convolution_13, squeeze_40, clamp_max_9, convolution_14, squeeze_43, add_76, convolution_15, squeeze_46, clamp_max_10, convolution_16, squeeze_49, clamp_max_11, convolution_17, squeeze_52, add_92, convolution_18, squeeze_55, clamp_max_12, convolution_19, squeeze_58, clamp_max_13, convolution_20, squeeze_61, add_107, convolution_21, squeeze_64, clamp_max_14, convolution_22, squeeze_67, clamp_max_15, convolution_23, squeeze_70, add_123, convolution_24, squeeze_73, clamp_max_16, convolution_25, squeeze_76, clamp_max_17, convolution_26, squeeze_79, add_139, convolution_27, squeeze_82, clamp_max_18, convolution_28, squeeze_85, clamp_max_19, convolution_29, squeeze_88, add_155, convolution_30, squeeze_91, clamp_max_20, convolution_31, squeeze_94, clamp_max_21, convolution_32, squeeze_97, add_170, convolution_33, squeeze_100, clamp_max_22, convolution_34, squeeze_103, clamp_max_23, convolution_35, squeeze_106, add_186, convolution_36, squeeze_109, clamp_max_24, convolution_37, squeeze_112, clamp_max_25, convolution_38, squeeze_115, add_202, convolution_39, squeeze_118, clamp_max_26, convolution_40, squeeze_121, clamp_max_27, convolution_41, squeeze_124, add_217, convolution_42, squeeze_127, clamp_max_28, convolution_43, squeeze_130, clamp_max_29, convolution_44, squeeze_133, add_233, convolution_45, squeeze_136, clamp_max_30, convolution_46, squeeze_139, clamp_max_31, convolution_47, squeeze_142, add_249, convolution_48, squeeze_145, clamp_max_32, convolution_49, squeeze_148, clamp_max_33, convolution_50, squeeze_151, add_264, convolution_51, squeeze_154, view, permute_1, bitwise_or, unsqueeze_210, unsqueeze_222, bitwise_or_1, unsqueeze_234, bitwise_or_2, unsqueeze_246, unsqueeze_258, bitwise_or_3, unsqueeze_270, bitwise_or_4, unsqueeze_282, unsqueeze_294, bitwise_or_5, unsqueeze_306, bitwise_or_6, unsqueeze_318, unsqueeze_330, bitwise_or_7, unsqueeze_342, bitwise_or_8, unsqueeze_354, unsqueeze_366, bitwise_or_9, unsqueeze_378, bitwise_or_10, unsqueeze_390, unsqueeze_402, bitwise_or_11, unsqueeze_414, bitwise_or_12, unsqueeze_426, unsqueeze_438, bitwise_or_13, unsqueeze_450, bitwise_or_14, unsqueeze_462, unsqueeze_474, bitwise_or_15, unsqueeze_486, bitwise_or_16, unsqueeze_498, unsqueeze_510, bitwise_or_17, unsqueeze_522, bitwise_or_18, unsqueeze_534, unsqueeze_546, bitwise_or_19, unsqueeze_558, bitwise_or_20, unsqueeze_570, unsqueeze_582, bitwise_or_21, unsqueeze_594, bitwise_or_22, unsqueeze_606, unsqueeze_618, bitwise_or_23, unsqueeze_630, bitwise_or_24, unsqueeze_642, unsqueeze_654, bitwise_or_25, unsqueeze_666, bitwise_or_26, unsqueeze_678, unsqueeze_690, bitwise_or_27, unsqueeze_702, bitwise_or_28, unsqueeze_714, unsqueeze_726, bitwise_or_29, unsqueeze_738, bitwise_or_30, unsqueeze_750, unsqueeze_762, bitwise_or_31, unsqueeze_774, bitwise_or_32, unsqueeze_786, unsqueeze_798, bitwise_or_33, unsqueeze_810, bitwise_or_34, unsqueeze_822, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv2_100', benchmark_compiled_module)
