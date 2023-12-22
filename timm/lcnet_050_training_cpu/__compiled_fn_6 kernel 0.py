
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


cpp_fused_convolution_backward_hardswish_backward_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(10240L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(-3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 < tmp2);
            auto tmp4 = static_cast<float>(3.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = to_float_mask(tmp0 <= tmp5);
            auto tmp8 = tmp0 / tmp5;
            auto tmp9 = static_cast<float>(0.5);
            auto tmp10 = at::vec::Vectorized<float>(tmp9);
            auto tmp11 = tmp8 + tmp10;
            auto tmp12 = tmp7 * tmp11;
            auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
            auto tmp14 = static_cast<float>(0.0);
            auto tmp15 = at::vec::Vectorized<float>(tmp14);
            auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
            tmp16.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_1 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x2) + (12544L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x2) + (12544L*x1)));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = static_cast<float>(49.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp0 / tmp5;
                        auto tmp12 = static_cast<float>(0.5);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 + tmp13;
                        auto tmp15 = tmp10 * tmp14;
                        auto tmp16 = decltype(tmp15)::blendv(tmp10, tmp15, tmp6);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = decltype(tmp18)::blendv(tmp16, tmp18, tmp3);
                        auto tmp22 = tmp20 - tmp21;
                        auto tmp23 = tmp19 * tmp22;
                        tmp_acc0_vec = tmp_acc0_vec + tmp19;
                        tmp_acc1_vec = tmp_acc1_vec + tmp23;
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                    auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = static_cast<float>(49.0);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 / tmp9;
                    auto tmp11 = tmp0 / tmp5;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 + tmp13;
                    auto tmp15 = tmp10 * tmp14;
                    auto tmp16 = decltype(tmp15)::blendv(tmp10, tmp15, tmp6);
                    auto tmp17 = static_cast<float>(0.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = decltype(tmp18)::blendv(tmp16, tmp18, tmp3);
                    auto tmp22 = tmp20 - tmp21;
                    auto tmp24 = static_cast<float>(0.002551020408163265);
                    auto tmp25 = at::vec::Vectorized<float>(tmp24);
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp28 = tmp27 * tmp27;
                    auto tmp29 = tmp26 * tmp28;
                    auto tmp30 = tmp22 * tmp29;
                    auto tmp31 = tmp19 - tmp30;
                    auto tmp33 = tmp32 * tmp25;
                    auto tmp34 = tmp31 - tmp33;
                    auto tmp36 = tmp27 * tmp35;
                    auto tmp37 = tmp34 * tmp36;
                    tmp37.store(out_ptr3 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (12544L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
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
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_4 = async_compile.cpp('''
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
    {
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x2) + (12544L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x2) + (12544L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x2) + (12544L*x1)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(49.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        auto tmp15 = tmp0 / tmp5;
                        auto tmp16 = static_cast<float>(0.5);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 + tmp17;
                        auto tmp19 = tmp14 * tmp18;
                        auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                        auto tmp26 = tmp24 - tmp25;
                        auto tmp27 = tmp23 * tmp26;
                        tmp_acc0_vec = tmp_acc0_vec + tmp23;
                        tmp_acc1_vec = tmp_acc1_vec + tmp27;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x0)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(49.0);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 / tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    auto tmp15 = tmp0 / tmp5;
                    auto tmp16 = static_cast<float>(0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 + tmp17;
                    auto tmp19 = tmp14 * tmp18;
                    auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                    auto tmp21 = static_cast<float>(0.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = static_cast<float>(0.002551020408163265);
                    auto tmp29 = at::vec::Vectorized<float>(tmp28);
                    auto tmp30 = tmp27 * tmp29;
                    auto tmp32 = tmp31 * tmp31;
                    auto tmp33 = tmp30 * tmp32;
                    auto tmp34 = tmp26 * tmp33;
                    auto tmp35 = tmp23 - tmp34;
                    auto tmp37 = tmp36 * tmp29;
                    auto tmp38 = tmp35 - tmp37;
                    tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_5 = async_compile.cpp('''
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp20 = tmp16 * tmp19;
                    tmp_acc0_vec = tmp_acc0_vec + tmp16;
                    tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(-3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 < tmp2);
                auto tmp4 = static_cast<float>(3.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = to_float_mask(tmp0 <= tmp5);
                auto tmp8 = tmp0 / tmp5;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = tmp7 * tmp11;
                auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                auto tmp14 = static_cast<float>(0.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                auto tmp19 = tmp17 - tmp18;
                auto tmp21 = static_cast<float>(0.002551020408163265);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp20 * tmp22;
                auto tmp25 = tmp24 * tmp24;
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp19 * tmp26;
                auto tmp28 = tmp16 - tmp27;
                auto tmp30 = tmp29 * tmp22;
                auto tmp31 = tmp28 - tmp30;
                auto tmp33 = tmp24 * tmp32;
                auto tmp34 = tmp31 * tmp33;
                tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (6272L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (6272L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_8 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x2) + (6272L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x2) + (6272L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x2) + (6272L*x1)));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(49.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        auto tmp15 = tmp0 / tmp5;
                        auto tmp16 = static_cast<float>(0.5);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 + tmp17;
                        auto tmp19 = tmp14 * tmp18;
                        auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                        auto tmp26 = tmp24 - tmp25;
                        auto tmp27 = tmp23 * tmp26;
                        tmp_acc0_vec = tmp_acc0_vec + tmp23;
                        tmp_acc1_vec = tmp_acc1_vec + tmp27;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (6272L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6272L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x0)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (6272L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = static_cast<float>(49.0);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 / tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    auto tmp15 = tmp0 / tmp5;
                    auto tmp16 = static_cast<float>(0.5);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 + tmp17;
                    auto tmp19 = tmp14 * tmp18;
                    auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                    auto tmp21 = static_cast<float>(0.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                    auto tmp26 = tmp24 - tmp25;
                    auto tmp28 = static_cast<float>(0.002551020408163265);
                    auto tmp29 = at::vec::Vectorized<float>(tmp28);
                    auto tmp30 = tmp27 * tmp29;
                    auto tmp32 = tmp31 * tmp31;
                    auto tmp33 = tmp30 * tmp32;
                    auto tmp34 = tmp26 * tmp33;
                    auto tmp35 = tmp23 - tmp34;
                    auto tmp37 = tmp36 * tmp29;
                    auto tmp38 = tmp35 - tmp37;
                    tmp38.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6272L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp4 = tmp0 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_9 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_10 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_11 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_12 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_13 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_14 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_15 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_16 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_17 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_19 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_20 = async_compile.cpp('''
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
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp20 = tmp16 * tmp19;
                    tmp_acc0_vec = tmp_acc0_vec + tmp16;
                    tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(-3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 < tmp2);
                auto tmp4 = static_cast<float>(3.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = to_float_mask(tmp0 <= tmp5);
                auto tmp8 = tmp0 / tmp5;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 + tmp10;
                auto tmp12 = tmp7 * tmp11;
                auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                auto tmp14 = static_cast<float>(0.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                auto tmp19 = tmp17 - tmp18;
                auto tmp21 = static_cast<float>(0.0006377551020408163);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp20 * tmp22;
                auto tmp25 = tmp24 * tmp24;
                auto tmp26 = tmp23 * tmp25;
                auto tmp27 = tmp19 * tmp26;
                auto tmp28 = tmp16 - tmp27;
                auto tmp30 = tmp29 * tmp22;
                auto tmp31 = tmp28 - tmp30;
                auto tmp33 = tmp24 * tmp32;
                auto tmp34 = tmp31 * tmp33;
                tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_21 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_22 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_23 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_24 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_25 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_26 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_28 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_29 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(9.964923469387754e-06);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (8L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (8L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(9.964923469387754e-06);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (8L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (8L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = static_cast<float>(9.964923469387754e-06);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp19 * tmp26;
                    auto tmp28 = tmp16 - tmp27;
                    auto tmp30 = tmp29 * tmp22;
                    auto tmp31 = tmp28 - tmp30;
                    auto tmp33 = tmp24 * tmp32;
                    auto tmp34 = tmp31 * tmp33;
                    tmp34.store(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_83, primals_85, primals_86, primals_87, primals_89, primals_91, primals_92, primals_175, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, clone_1, div_1, convolution_2, squeeze_7, clone_2, div_2, convolution_3, squeeze_10, clone_3, div_3, convolution_4, squeeze_13, clone_4, div_4, convolution_5, squeeze_16, clone_5, div_5, convolution_6, squeeze_19, clone_6, div_6, convolution_7, squeeze_22, clone_7, div_7, convolution_8, squeeze_25, clone_8, div_8, convolution_9, squeeze_28, clone_9, div_9, convolution_10, squeeze_31, clone_10, div_10, convolution_11, squeeze_34, clone_11, div_11, convolution_12, squeeze_37, clone_12, div_12, convolution_13, squeeze_40, clone_13, div_13, convolution_14, squeeze_43, clone_14, div_14, convolution_15, squeeze_46, clone_15, div_15, convolution_16, squeeze_49, clone_16, div_16, convolution_17, squeeze_52, clone_17, div_17, convolution_18, squeeze_55, clone_18, div_18, convolution_19, squeeze_58, clone_19, div_19, convolution_20, squeeze_61, clone_20, div_20, convolution_21, squeeze_64, clone_21, div_21, convolution_22, squeeze_67, clone_22, div_22, convolution_23, squeeze_70, clone_23, div_23, mean, relu, div_24, mul_192, convolution_26, squeeze_73, clone_24, div_25, convolution_27, squeeze_76, clone_25, div_26, mean_1, relu_1, div_27, mul_209, convolution_30, squeeze_79, clone_26, mean_2, convolution_31, view_1, permute_1, unsqueeze_110, bitwise_and, unsqueeze_122, unsqueeze_134, bitwise_and_1, unsqueeze_146, unsqueeze_158, unsqueeze_170, unsqueeze_182, unsqueeze_194, unsqueeze_206, unsqueeze_218, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (8, ), (1, ))
    assert_size_stride(primals_3, (8, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_57, (8, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_58, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_59, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_60, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_61, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_62, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_63, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_64, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_65, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_66, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_67, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_68, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_69, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_70, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_71, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_72, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_73, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_74, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_75, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_76, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_77, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_78, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_79, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_80, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_81, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_83, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_85, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_86, (256, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_87, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_89, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_91, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (1280, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_175, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(squeeze_1, (8, ), (1, ))
    assert_size_stride(clone, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(div, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(convolution_1, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(squeeze_4, (8, ), (1, ))
    assert_size_stride(clone_1, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(div_1, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(clone_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 16, 56, 56), (50176, 1, 896, 16))
    assert_size_stride(squeeze_10, (16, ), (1, ))
    assert_size_stride(clone_3, (8, 16, 56, 56), (50176, 1, 896, 16))
    assert_size_stride(div_3, (8, 16, 56, 56), (50176, 1, 896, 16))
    assert_size_stride(convolution_4, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_13, (32, ), (1, ))
    assert_size_stride(clone_4, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(div_4, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_5, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_16, (32, ), (1, ))
    assert_size_stride(clone_5, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(div_5, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_6, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(squeeze_19, (32, ), (1, ))
    assert_size_stride(clone_6, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(div_6, (8, 32, 56, 56), (100352, 1, 1792, 32))
    assert_size_stride(convolution_7, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(squeeze_22, (32, ), (1, ))
    assert_size_stride(clone_7, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(div_7, (8, 32, 28, 28), (25088, 1, 896, 32))
    assert_size_stride(convolution_8, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(clone_8, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(div_8, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_9, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_28, (64, ), (1, ))
    assert_size_stride(clone_9, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(div_9, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_10, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(squeeze_31, (64, ), (1, ))
    assert_size_stride(clone_10, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(div_10, (8, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(convolution_11, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(squeeze_34, (64, ), (1, ))
    assert_size_stride(clone_11, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(div_11, (8, 64, 14, 14), (12544, 1, 896, 64))
    assert_size_stride(convolution_12, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(clone_12, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_12, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_13, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_40, (128, ), (1, ))
    assert_size_stride(clone_13, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_13, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_14, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_43, (128, ), (1, ))
    assert_size_stride(clone_14, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_14, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_15, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(clone_15, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_15, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_16, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_49, (128, ), (1, ))
    assert_size_stride(clone_16, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_16, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_17, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_52, (128, ), (1, ))
    assert_size_stride(clone_17, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_17, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_18, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_55, (128, ), (1, ))
    assert_size_stride(clone_18, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_18, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_19, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(clone_19, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_19, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_20, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_61, (128, ), (1, ))
    assert_size_stride(clone_20, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_20, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_21, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(clone_21, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_21, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_22, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(clone_22, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(div_22, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_23, (8, 128, 7, 7), (6272, 1, 896, 128))
    assert_size_stride(squeeze_70, (128, ), (1, ))
    assert_size_stride(clone_23, (8, 128, 7, 7), (6272, 1, 896, 128))
    assert_size_stride(div_23, (8, 128, 7, 7), (6272, 1, 896, 128))
    assert_size_stride(mean, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(relu, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_24, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(mul_192, (8, 128, 7, 7), (6272, 1, 896, 128))
    assert_size_stride(convolution_26, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(clone_24, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(div_25, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(convolution_27, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_76, (256, ), (1, ))
    assert_size_stride(clone_25, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(div_26, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(mean_1, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(relu_1, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(div_27, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_209, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(convolution_30, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(squeeze_79, (256, ), (1, ))
    assert_size_stride(clone_26, (8, 256, 7, 7), (12544, 1, 1792, 256))
    assert_size_stride(mean_2, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_31, (8, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(view_1, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(unsqueeze_110, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(bitwise_and, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(unsqueeze_122, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_134, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(bitwise_and_1, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(unsqueeze_146, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_158, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_170, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_182, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_194, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_206, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_218, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_230, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_242, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_254, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_266, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_278, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_290, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_302, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_314, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_326, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_338, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_350, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_374, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_398, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_410, (1, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(unsqueeze_422, (1, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view_1, out=buf1)
    del view_1
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = reinterpret_tensor(buf0, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf0  # reuse
    cpp_fused_convolution_backward_hardswish_backward_sum_0(c_void_p(buf3.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(buf2.data_ptr()))
    del convolution_31
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward]
    buf4 = aten.convolution_backward(buf3, mean_2, primals_92, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf3
    del mean_2
    del primals_92
    buf5 = buf4[0]
    buf6 = buf4[1]
    buf7 = buf4[2]
    del buf4
    buf8 = empty((256, ), device='cpu', dtype=torch.float32)
    buf9 = empty((256, ), device='cpu', dtype=torch.float32)
    buf10 = empty((256, ), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_1(c_void_p(clone_26.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_110.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    del clone_26
    del convolution_30
    del primals_53
    del squeeze_79
    del unsqueeze_110
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf12 = aten.convolution_backward(buf11, mul_209, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf11
    del mul_209
    del primals_91
    buf13 = buf12[0]
    buf14 = buf12[1]
    del buf12
    buf15 = reinterpret_tensor(buf5, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf5  # reuse
    buf16 = reinterpret_tensor(buf15, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf15  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_2(c_void_p(buf16.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(bitwise_and.data_ptr()))
    del bitwise_and
    del div_26
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf17 = aten.convolution_backward(buf16, relu_1, primals_89, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf16
    del primals_89
    buf18 = buf17[0]
    buf19 = buf17[1]
    buf20 = buf17[2]
    del buf17
    buf21 = buf18; del buf18  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_3(c_void_p(buf21.data_ptr()), c_void_p(relu_1.data_ptr()))
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf22 = aten.convolution_backward(buf21, mean_1, primals_87, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf21
    del mean_1
    del primals_87
    buf23 = buf22[0]
    buf24 = buf22[1]
    buf25 = buf22[2]
    del buf22
    buf26 = buf9; del buf9  # reuse
    buf27 = empty((256, ), device='cpu', dtype=torch.float32)
    buf28 = buf13; del buf13  # reuse
    buf29 = buf27; del buf27  # reuse
    buf30 = buf28; del buf28  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_4(c_void_p(buf30.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(clone_25.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_122.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf26.data_ptr()))
    del buf23
    del clone_25
    del convolution_27
    del div_27
    del primals_51
    del squeeze_76
    del unsqueeze_122
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf31 = aten.convolution_backward(buf30, div_25, primals_86, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 256, [True, True, False])
    del buf30
    del div_25
    del primals_86
    buf32 = buf31[0]
    buf33 = buf31[1]
    del buf31
    buf34 = empty((256, ), device='cpu', dtype=torch.float32)
    buf35 = empty((256, ), device='cpu', dtype=torch.float32)
    buf36 = empty((256, ), device='cpu', dtype=torch.float32)
    buf37 = buf32; del buf32  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_5(c_void_p(buf37.data_ptr()), c_void_p(clone_24.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_134.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del buf35
    del clone_24
    del convolution_26
    del primals_49
    del squeeze_73
    del unsqueeze_134
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf38 = aten.convolution_backward(buf37, mul_192, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf37
    del mul_192
    del primals_85
    buf39 = buf38[0]
    buf40 = buf38[1]
    del buf38
    buf41 = empty_strided((8, 128, 1, 1), (128, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf42 = reinterpret_tensor(buf41, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf41  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_6(c_void_p(buf42.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(bitwise_and_1.data_ptr()))
    del bitwise_and_1
    del div_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf43 = aten.convolution_backward(buf42, relu, primals_83, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf42
    del primals_83
    buf44 = buf43[0]
    buf45 = buf43[1]
    buf46 = buf43[2]
    del buf43
    buf47 = buf44; del buf44  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_7(c_void_p(buf47.data_ptr()), c_void_p(relu.data_ptr()))
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf48 = aten.convolution_backward(buf47, mean, primals_81, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf47
    del mean
    del primals_81
    buf49 = buf48[0]
    buf50 = buf48[1]
    buf51 = buf48[2]
    del buf48
    buf52 = empty((128, ), device='cpu', dtype=torch.float32)
    buf53 = empty((128, ), device='cpu', dtype=torch.float32)
    buf54 = buf39; del buf39  # reuse
    buf55 = buf53; del buf53  # reuse
    buf56 = buf54; del buf54  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_8(c_void_p(buf56.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(clone_23.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_146.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf52.data_ptr()))
    del buf49
    del clone_23
    del convolution_23
    del div_24
    del primals_47
    del squeeze_70
    del unsqueeze_146
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf57 = aten.convolution_backward(buf56, div_22, primals_80, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf56
    del div_22
    del primals_80
    buf58 = buf57[0]
    buf59 = buf57[1]
    del buf57
    buf60 = empty((128, ), device='cpu', dtype=torch.float32)
    buf61 = empty((128, ), device='cpu', dtype=torch.float32)
    buf62 = empty((128, ), device='cpu', dtype=torch.float32)
    buf63 = buf58; del buf58  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_9(c_void_p(buf63.data_ptr()), c_void_p(clone_22.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_158.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del clone_22
    del convolution_22
    del primals_45
    del squeeze_67
    del unsqueeze_158
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf64 = aten.convolution_backward(buf63, div_21, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf63
    del div_21
    del primals_79
    buf65 = buf64[0]
    buf66 = buf64[1]
    del buf64
    buf67 = buf61; del buf61  # reuse
    buf68 = empty((128, ), device='cpu', dtype=torch.float32)
    buf69 = empty((128, ), device='cpu', dtype=torch.float32)
    buf70 = buf65; del buf65  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_10(c_void_p(buf70.data_ptr()), c_void_p(clone_21.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_170.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del clone_21
    del convolution_21
    del primals_43
    del squeeze_64
    del unsqueeze_170
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf71 = aten.convolution_backward(buf70, div_20, primals_78, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf70
    del div_20
    del primals_78
    buf72 = buf71[0]
    buf73 = buf71[1]
    del buf71
    buf74 = buf68; del buf68  # reuse
    buf75 = empty((128, ), device='cpu', dtype=torch.float32)
    buf76 = empty((128, ), device='cpu', dtype=torch.float32)
    buf77 = buf72; del buf72  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_11(c_void_p(buf77.data_ptr()), c_void_p(clone_20.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_182.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    del clone_20
    del convolution_20
    del primals_41
    del squeeze_61
    del unsqueeze_182
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf78 = aten.convolution_backward(buf77, div_19, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf77
    del div_19
    del primals_77
    buf79 = buf78[0]
    buf80 = buf78[1]
    del buf78
    buf81 = buf75; del buf75  # reuse
    buf82 = empty((128, ), device='cpu', dtype=torch.float32)
    buf83 = empty((128, ), device='cpu', dtype=torch.float32)
    buf84 = buf79; del buf79  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_12(c_void_p(buf84.data_ptr()), c_void_p(clone_19.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_194.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del clone_19
    del convolution_19
    del primals_39
    del squeeze_58
    del unsqueeze_194
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf85 = aten.convolution_backward(buf84, div_18, primals_76, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf84
    del div_18
    del primals_76
    buf86 = buf85[0]
    buf87 = buf85[1]
    del buf85
    buf88 = buf82; del buf82  # reuse
    buf89 = empty((128, ), device='cpu', dtype=torch.float32)
    buf90 = empty((128, ), device='cpu', dtype=torch.float32)
    buf91 = buf86; del buf86  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_13(c_void_p(buf91.data_ptr()), c_void_p(clone_18.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_206.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del clone_18
    del convolution_18
    del primals_37
    del squeeze_55
    del unsqueeze_206
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf92 = aten.convolution_backward(buf91, div_17, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf91
    del div_17
    del primals_75
    buf93 = buf92[0]
    buf94 = buf92[1]
    del buf92
    buf95 = buf89; del buf89  # reuse
    buf96 = empty((128, ), device='cpu', dtype=torch.float32)
    buf97 = empty((128, ), device='cpu', dtype=torch.float32)
    buf98 = buf93; del buf93  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_14(c_void_p(buf98.data_ptr()), c_void_p(clone_17.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_218.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del clone_17
    del convolution_17
    del primals_35
    del squeeze_52
    del unsqueeze_218
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf99 = aten.convolution_backward(buf98, div_16, primals_74, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf98
    del div_16
    del primals_74
    buf100 = buf99[0]
    buf101 = buf99[1]
    del buf99
    buf102 = buf96; del buf96  # reuse
    buf103 = empty((128, ), device='cpu', dtype=torch.float32)
    buf104 = empty((128, ), device='cpu', dtype=torch.float32)
    buf105 = buf100; del buf100  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_15(c_void_p(buf105.data_ptr()), c_void_p(clone_16.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_230.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    del clone_16
    del convolution_16
    del primals_33
    del squeeze_49
    del unsqueeze_230
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf106 = aten.convolution_backward(buf105, div_15, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf105
    del div_15
    del primals_73
    buf107 = buf106[0]
    buf108 = buf106[1]
    del buf106
    buf109 = buf103; del buf103  # reuse
    buf110 = empty((128, ), device='cpu', dtype=torch.float32)
    buf111 = empty((128, ), device='cpu', dtype=torch.float32)
    buf112 = buf107; del buf107  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_16(c_void_p(buf112.data_ptr()), c_void_p(clone_15.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_242.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    del clone_15
    del convolution_15
    del primals_31
    del squeeze_46
    del unsqueeze_242
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf113 = aten.convolution_backward(buf112, div_14, primals_72, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf112
    del div_14
    del primals_72
    buf114 = buf113[0]
    buf115 = buf113[1]
    del buf113
    buf116 = buf110; del buf110  # reuse
    buf117 = empty((128, ), device='cpu', dtype=torch.float32)
    buf118 = empty((128, ), device='cpu', dtype=torch.float32)
    buf119 = buf114; del buf114  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_17(c_void_p(buf119.data_ptr()), c_void_p(clone_14.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_254.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()))
    del clone_14
    del convolution_14
    del primals_29
    del squeeze_43
    del unsqueeze_254
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf120 = aten.convolution_backward(buf119, div_13, primals_71, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf119
    del div_13
    del primals_71
    buf121 = buf120[0]
    buf122 = buf120[1]
    del buf120
    buf123 = buf117; del buf117  # reuse
    buf124 = empty((128, ), device='cpu', dtype=torch.float32)
    buf125 = empty((128, ), device='cpu', dtype=torch.float32)
    buf126 = buf121; del buf121  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18(c_void_p(buf126.data_ptr()), c_void_p(clone_13.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_266.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    del clone_13
    del convolution_13
    del primals_27
    del squeeze_40
    del unsqueeze_266
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf127 = aten.convolution_backward(buf126, div_12, primals_70, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False])
    del buf126
    del div_12
    del primals_70
    buf128 = buf127[0]
    buf129 = buf127[1]
    del buf127
    buf130 = buf124; del buf124  # reuse
    buf131 = empty((128, ), device='cpu', dtype=torch.float32)
    buf132 = empty((128, ), device='cpu', dtype=torch.float32)
    buf133 = buf128; del buf128  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_19(c_void_p(buf133.data_ptr()), c_void_p(clone_12.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_278.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del buf131
    del clone_12
    del convolution_12
    del primals_25
    del squeeze_37
    del unsqueeze_278
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf134 = aten.convolution_backward(buf133, div_11, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf133
    del div_11
    del primals_69
    buf135 = buf134[0]
    buf136 = buf134[1]
    del buf134
    buf137 = empty((64, ), device='cpu', dtype=torch.float32)
    buf138 = empty((64, ), device='cpu', dtype=torch.float32)
    buf139 = empty((64, ), device='cpu', dtype=torch.float32)
    buf140 = buf135; del buf135  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_20(c_void_p(buf140.data_ptr()), c_void_p(clone_11.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_290.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del clone_11
    del convolution_11
    del primals_23
    del squeeze_34
    del unsqueeze_290
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf141 = aten.convolution_backward(buf140, div_10, primals_68, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
    del buf140
    del div_10
    del primals_68
    buf142 = buf141[0]
    buf143 = buf141[1]
    del buf141
    buf144 = buf138; del buf138  # reuse
    buf145 = empty((64, ), device='cpu', dtype=torch.float32)
    buf146 = empty((64, ), device='cpu', dtype=torch.float32)
    buf147 = buf142; del buf142  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_21(c_void_p(buf147.data_ptr()), c_void_p(clone_10.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_302.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    del clone_10
    del convolution_10
    del primals_21
    del squeeze_31
    del unsqueeze_302
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf148 = aten.convolution_backward(buf147, div_9, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf147
    del div_9
    del primals_67
    buf149 = buf148[0]
    buf150 = buf148[1]
    del buf148
    buf151 = buf145; del buf145  # reuse
    buf152 = empty((64, ), device='cpu', dtype=torch.float32)
    buf153 = empty((64, ), device='cpu', dtype=torch.float32)
    buf154 = buf149; del buf149  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_22(c_void_p(buf154.data_ptr()), c_void_p(clone_9.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_314.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    del clone_9
    del convolution_9
    del primals_19
    del squeeze_28
    del unsqueeze_314
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf155 = aten.convolution_backward(buf154, div_8, primals_66, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
    del buf154
    del div_8
    del primals_66
    buf156 = buf155[0]
    buf157 = buf155[1]
    del buf155
    buf158 = buf152; del buf152  # reuse
    buf159 = empty((64, ), device='cpu', dtype=torch.float32)
    buf160 = empty((64, ), device='cpu', dtype=torch.float32)
    buf161 = buf156; del buf156  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_23(c_void_p(buf161.data_ptr()), c_void_p(clone_8.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_326.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del buf159
    del clone_8
    del convolution_8
    del primals_17
    del squeeze_25
    del unsqueeze_326
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf162 = aten.convolution_backward(buf161, div_7, primals_65, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf161
    del div_7
    del primals_65
    buf163 = buf162[0]
    buf164 = buf162[1]
    del buf162
    buf165 = empty((32, ), device='cpu', dtype=torch.float32)
    buf166 = empty((32, ), device='cpu', dtype=torch.float32)
    buf167 = empty((32, ), device='cpu', dtype=torch.float32)
    buf168 = buf163; del buf163  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_24(c_void_p(buf168.data_ptr()), c_void_p(clone_7.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_338.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    del clone_7
    del convolution_7
    del primals_15
    del squeeze_22
    del unsqueeze_338
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf169 = aten.convolution_backward(buf168, div_6, primals_64, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf168
    del div_6
    del primals_64
    buf170 = buf169[0]
    buf171 = buf169[1]
    del buf169
    buf172 = buf166; del buf166  # reuse
    buf173 = empty((32, ), device='cpu', dtype=torch.float32)
    buf174 = empty((32, ), device='cpu', dtype=torch.float32)
    buf175 = buf170; del buf170  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_25(c_void_p(buf175.data_ptr()), c_void_p(clone_6.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_350.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del clone_6
    del convolution_6
    del primals_13
    del squeeze_19
    del unsqueeze_350
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf176 = aten.convolution_backward(buf175, div_5, primals_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf175
    del div_5
    del primals_63
    buf177 = buf176[0]
    buf178 = buf176[1]
    del buf176
    buf179 = buf173; del buf173  # reuse
    buf180 = empty((32, ), device='cpu', dtype=torch.float32)
    buf181 = empty((32, ), device='cpu', dtype=torch.float32)
    buf182 = buf177; del buf177  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_26(c_void_p(buf182.data_ptr()), c_void_p(clone_5.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_362.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del clone_5
    del convolution_5
    del primals_11
    del squeeze_16
    del unsqueeze_362
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf183 = aten.convolution_backward(buf182, div_4, primals_62, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf182
    del div_4
    del primals_62
    buf184 = buf183[0]
    buf185 = buf183[1]
    del buf183
    buf186 = buf180; del buf180  # reuse
    buf187 = empty((32, ), device='cpu', dtype=torch.float32)
    buf188 = empty((32, ), device='cpu', dtype=torch.float32)
    buf189 = buf184; del buf184  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_27(c_void_p(buf189.data_ptr()), c_void_p(clone_4.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_374.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    del buf187
    del clone_4
    del convolution_4
    del primals_9
    del squeeze_13
    del unsqueeze_374
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf190 = aten.convolution_backward(buf189, div_3, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf189
    del div_3
    del primals_61
    buf191 = buf190[0]
    buf192 = buf190[1]
    del buf190
    buf193 = empty((16, ), device='cpu', dtype=torch.float32)
    buf194 = empty((16, ), device='cpu', dtype=torch.float32)
    buf195 = empty((16, ), device='cpu', dtype=torch.float32)
    buf196 = buf191; del buf191  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_28(c_void_p(buf196.data_ptr()), c_void_p(clone_3.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_386.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    del clone_3
    del convolution_3
    del primals_7
    del squeeze_10
    del unsqueeze_386
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf197 = aten.convolution_backward(buf196, div_2, primals_60, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
    del buf196
    del div_2
    del primals_60
    buf198 = buf197[0]
    buf199 = buf197[1]
    del buf197
    buf200 = buf194; del buf194  # reuse
    buf201 = empty((16, ), device='cpu', dtype=torch.float32)
    buf202 = empty((16, ), device='cpu', dtype=torch.float32)
    buf203 = buf198; del buf198  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_29(c_void_p(buf203.data_ptr()), c_void_p(clone_2.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_398.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    del buf201
    del clone_2
    del convolution_2
    del primals_5
    del squeeze_7
    del unsqueeze_398
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf204 = aten.convolution_backward(buf203, div_1, primals_59, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf203
    del div_1
    del primals_59
    buf205 = buf204[0]
    buf206 = buf204[1]
    del buf204
    buf207 = empty((8, ), device='cpu', dtype=torch.float32)
    buf208 = empty((8, ), device='cpu', dtype=torch.float32)
    buf209 = empty((8, ), device='cpu', dtype=torch.float32)
    buf210 = buf205; del buf205  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_30(c_void_p(buf210.data_ptr()), c_void_p(clone_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_410.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    del clone_1
    del convolution_1
    del primals_3
    del squeeze_4
    del unsqueeze_410
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf211 = aten.convolution_backward(buf210, div, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf210
    del div
    del primals_58
    buf212 = buf211[0]
    buf213 = buf211[1]
    del buf211
    buf214 = buf208; del buf208  # reuse
    buf215 = empty((8, ), device='cpu', dtype=torch.float32)
    buf216 = empty((8, ), device='cpu', dtype=torch.float32)
    buf217 = buf212; del buf212  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_31(c_void_p(buf217.data_ptr()), c_void_p(clone.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_422.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del buf215
    del clone
    del convolution
    del primals_1
    del squeeze_1
    del unsqueeze_422
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf218 = aten.convolution_backward(buf217, primals_175, primals_57, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf217
    del primals_175
    del primals_57
    buf219 = buf218[1]
    return (buf216, buf214, buf209, buf207, buf202, buf200, buf195, buf193, buf188, buf186, buf181, buf179, buf174, buf172, buf167, buf165, buf160, buf158, buf153, buf151, buf146, buf144, buf139, buf137, buf132, buf130, buf125, buf123, buf118, buf116, buf111, buf109, buf104, buf102, buf97, buf95, buf90, buf88, buf83, buf81, buf76, buf74, buf69, buf67, buf62, buf60, buf55, buf52, buf36, buf34, buf29, buf26, buf10, buf8, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), buf219, buf213, buf206, buf199, buf192, buf185, buf178, buf171, buf164, buf157, buf150, buf143, buf136, buf129, buf122, buf115, buf108, buf101, buf94, buf87, buf80, buf73, buf66, buf59, buf50, buf51, buf45, buf46, buf40, buf33, buf24, buf25, buf19, buf20, buf14, buf6, buf7, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((8, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((256, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1280, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    clone_1 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    clone_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    clone_3 = rand_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    clone_4 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    clone_5 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    clone_6 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    clone_7 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    clone_8 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    clone_9 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    clone_10 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    clone_11 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_12 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_13 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_14 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_15 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_16 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_17 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_18 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_19 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_20 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_21 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_22 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    clone_23 = rand_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    mul_192 = rand_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    clone_24 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    clone_25 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    mul_209 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    clone_26 = rand_strided((8, 256, 7, 7), (12544, 1, 1792, 256), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 1280, 1, 1), (1280, 1, 1280, 1280), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((8, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    unsqueeze_110 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.bool)
    unsqueeze_122 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_134 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.bool)
    unsqueeze_146 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_158 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_170 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_182 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_194 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_206 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_218 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_230 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_242 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_254 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_266 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_278 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_290 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_302 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_314 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_326 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_338 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_350 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_374 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_83, primals_85, primals_86, primals_87, primals_89, primals_91, primals_92, primals_175, convolution, squeeze_1, clone, div, convolution_1, squeeze_4, clone_1, div_1, convolution_2, squeeze_7, clone_2, div_2, convolution_3, squeeze_10, clone_3, div_3, convolution_4, squeeze_13, clone_4, div_4, convolution_5, squeeze_16, clone_5, div_5, convolution_6, squeeze_19, clone_6, div_6, convolution_7, squeeze_22, clone_7, div_7, convolution_8, squeeze_25, clone_8, div_8, convolution_9, squeeze_28, clone_9, div_9, convolution_10, squeeze_31, clone_10, div_10, convolution_11, squeeze_34, clone_11, div_11, convolution_12, squeeze_37, clone_12, div_12, convolution_13, squeeze_40, clone_13, div_13, convolution_14, squeeze_43, clone_14, div_14, convolution_15, squeeze_46, clone_15, div_15, convolution_16, squeeze_49, clone_16, div_16, convolution_17, squeeze_52, clone_17, div_17, convolution_18, squeeze_55, clone_18, div_18, convolution_19, squeeze_58, clone_19, div_19, convolution_20, squeeze_61, clone_20, div_20, convolution_21, squeeze_64, clone_21, div_21, convolution_22, squeeze_67, clone_22, div_22, convolution_23, squeeze_70, clone_23, div_23, mean, relu, div_24, mul_192, convolution_26, squeeze_73, clone_24, div_25, convolution_27, squeeze_76, clone_25, div_26, mean_1, relu_1, div_27, mul_209, convolution_30, squeeze_79, clone_26, mean_2, convolution_31, view_1, permute_1, unsqueeze_110, bitwise_and, unsqueeze_122, unsqueeze_134, bitwise_and_1, unsqueeze_146, unsqueeze_158, unsqueeze_170, unsqueeze_182, unsqueeze_194, unsqueeze_206, unsqueeze_218, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('lcnet_050', benchmark_compiled_module)
