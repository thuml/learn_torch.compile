
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


cpp_fused_convolution_backward_native_batch_norm_backward_sum_threshold_backward_0 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (192L*x2) + (12288L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1856L + x0 + (2048L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x2) + (12288L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(64.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr3 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1856L + x2 + (2048L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                    auto tmp2 = static_cast<float>(64.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
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
                    tmp25.store(out_ptr4 + static_cast<long>(x2 + (192L*x1) + (12288L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (2048L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(8L, 2L + x3))))) + (16384L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(8L, 2L + x2))))) + (131072L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (2048L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (16384L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(8L, 2L + x2))))) + (131072L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (2048L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (16384L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(8L, 2L + x2))))) + (131072L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (2048L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(8L, 2L + x3))))) + (16384L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (131072L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(x1 + (2048L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (16384L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (131072L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(x1 + (2048L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (16384L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (131072L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(x1 + (2048L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(8L, 2L + x3))))) + (16384L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (131072L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(x1 + (2048L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (16384L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (131072L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(x1 + (2048L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (16384L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (131072L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(8L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(8L, 2L + x3));
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
                            out_ptr0[static_cast<long>(x3 + (8L*x2) + (64L*x1) + (131072L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (384L*x2) + (24576L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1472L + x0 + (2048L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x2) + (24576L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(64.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1472L + x2 + (2048L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(64.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
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
                        tmp25.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (24576L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1088L + x0 + (2048L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (24576L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(64.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1088L + x2 + (2048L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(64.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
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
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (24576L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(704L + x0 + (2048L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (24576L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(64.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(704L + x2 + (2048L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(64.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
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
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (24576L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(320L + x0 + (2048L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (24576L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(64.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(320L + x2 + (2048L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(64.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
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
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (384L*x1) + (24576L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (320L*x2) + (20480L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (320L*x2) + (20480L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(64.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
                {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (320L*x1) + (20480L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (320L*x1) + (20480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(64.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
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
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (320L*x1) + (20480L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(118784L + x1 + (64L*x2) + (131072L*x0)), static_cast<long>(64L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (12288L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1856L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1856L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1856L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                        tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (12288L*x0)));
                    }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr3 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (1280L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(8L, 2L + x3))))) + (10240L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(8L, 2L + x2))))) + (81920L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (1280L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (10240L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(8L, 2L + x2))))) + (81920L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (1280L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (10240L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(8L, 2L + x2))))) + (81920L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (1280L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(8L, 2L + x3))))) + (10240L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (81920L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(x1 + (1280L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (10240L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (81920L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(x1 + (1280L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (10240L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (81920L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(x1 + (1280L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(8L, 2L + x3))))) + (10240L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (81920L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(x1 + (1280L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (10240L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (81920L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(x1 + (1280L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(8L, 2L + x3))))) + (10240L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(8L, 2L + x2))))) + (81920L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(8L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(8L, 2L + x3));
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
                            out_ptr0[static_cast<long>(x3 + (8L*x2) + (64L*x1) + (81920L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(94208L + x1 + (64L*x2) + (131072L*x0)), static_cast<long>(64L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (24576L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1472L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1472L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(1472L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr1 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (24576L*x0)));
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(69632L + x1 + (64L*x2) + (131072L*x0)), static_cast<long>(64L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (24576L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1088L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1088L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1088L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (24576L*x0)));
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (448L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(45056L + x1 + (64L*x2) + (131072L*x0)), static_cast<long>(64L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (24576L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(704L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(704L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(704L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (24576L*x0)));
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(20480L + x1 + (64L*x2) + (131072L*x0)), static_cast<long>(64L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (24576L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(320L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(320L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(320L + x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (24576L*x0)));
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (131072L*x0)), static_cast<long>(64L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (320L*x1) + (320L*x1_inner) + (20480L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (320L*x1) + (320L*x1_inner) + (20480L*x0)));
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (320L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (320L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (320L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (64L*x2) + (81920L*x0)), static_cast<long>(64L), tmp0, 8);
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (81920L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (81920L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (81920L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp8, 8, in_out_ptr0 + static_cast<long>(x1 + (64L*x2) + (81920L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(20480L + x2 + (64L*x0) + (81920L*x1)), static_cast<long>(64L), tmp1, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(20480L + x2 + (64L*x0) + (81920L*x1)), static_cast<long>(64L), tmp1, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x2) + (192L*x2_inner) + (12288L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x2) + (192L*x2_inner) + (12288L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp9 = tmp5 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(20480L + x1 + (64L*x2) + (81920L*x0)), static_cast<long>(64L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (12288L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (12288L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp3 = static_cast<float>(0.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
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
                        tmp23.store(out_ptr3 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (12288L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (81920L*x1)), static_cast<long>(64L), tmp1, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (81920L*x1)), static_cast<long>(64L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (320L*x2) + (320L*x2_inner) + (20480L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (320L*x2) + (320L*x2_inner) + (20480L*x1)));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = static_cast<float>(0.0);
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                                auto tmp8 = tmp6 - tmp7;
                                auto tmp9 = tmp5 * tmp8;
                                tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                tmp_acc1_vec = tmp_acc1_vec + tmp9;
                            }
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
                {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (81920L*x0)), static_cast<long>(64L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (320L*x1) + (320L*x1_inner) + (20480L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (320L*x1) + (320L*x1_inner) + (20480L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
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
                            tmp23.store(out_ptr3 + static_cast<long>(x2 + (320L*x1) + (320L*x1_inner) + (20480L*x0)));
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(576L + x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(576L + x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(576L + x0 + (768L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(576L + x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(576L + x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(576L + x1 + (768L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00043252595155709344);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_26 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(17L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(17L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(17L, 2L + x3));
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
                            out_ptr0[static_cast<long>(x3 + (17L*x2) + (289L*x1) + (221952L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(384L + x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(384L + x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(384L + x0 + (768L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(384L + x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(384L + x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(384L + x1 + (768L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00043252595155709344);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(192L + x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(192L + x0 + (768L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(192L + x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(192L + x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(192L + x1 + (768L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00043252595155709344);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00043252595155709344);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(166464L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(576L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(576L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(576L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(166464L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(576L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(576L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(576L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(17L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(17L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(17L, 2L + x3));
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
                            out_ptr0[static_cast<long>(x3 + (17L*x2) + (289L*x1) + (221952L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(110976L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(384L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(384L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(384L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr1 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(110976L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr3[static_cast<long>(384L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr4[static_cast<long>(384L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr5[static_cast<long>(384L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr1[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(55488L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(192L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(192L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(192L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(55488L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(192L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(192L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(192L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(166464L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(576L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(576L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(576L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(166464L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(576L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(576L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(576L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(17L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(17L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(17L, 2L + x3));
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
                            out_ptr0[static_cast<long>(x3 + (17L*x2) + (289L*x1) + (221952L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(110976L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(384L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(384L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(384L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr1 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(110976L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr3[static_cast<long>(384L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr4[static_cast<long>(384L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr5[static_cast<long>(384L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr1[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(55488L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(192L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(192L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(192L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(55488L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(192L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(192L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(192L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(166464L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(576L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(576L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(576L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(166464L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(576L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(576L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(576L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(17L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(x1 + (768L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(17L, 2L + x3))))) + (13056L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(17L, 2L + x2))))) + (221952L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(17L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(17L, 2L + x3));
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
                            out_ptr0[static_cast<long>(x3 + (17L*x2) + (289L*x1) + (221952L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(110976L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(384L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(384L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(384L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr1 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(110976L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr3[static_cast<long>(384L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr4[static_cast<long>(384L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr5[static_cast<long>(384L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr1[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(55488L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(192L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(192L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(192L + x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(55488L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(192L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(192L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(192L + x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00043252595155709344);
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (55488L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (289L*x2) + (221952L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (192L*x1) + (55488L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2312L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00043252595155709344);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp0, 8);
                        float tmp8[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (221952L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp8, 8, in_out_ptr0 + static_cast<long>(x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (289L*x2) + (221952L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (221952L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        in_out_ptr0[static_cast<long>(x1 + (289L*x2) + (221952L*x0))] = tmp6;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(288L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(110976L + x2 + (289L*x0) + (221952L*x1)), static_cast<long>(289L), tmp1, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(110976L + x2 + (289L*x0) + (221952L*x1)), static_cast<long>(289L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (96L*x2) + (96L*x2_inner) + (27744L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x2) + (96L*x2_inner) + (27744L*x1)));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = static_cast<float>(0.0);
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                                auto tmp8 = tmp6 - tmp7;
                                auto tmp9 = tmp5 * tmp8;
                                tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                tmp_acc1_vec = tmp_acc1_vec + tmp9;
                            }
                        }
                        for(long x2=static_cast<long>(288L); x2<static_cast<long>(289L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (96L*x2) + (27744L*x1)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(110976L + x2 + (289L*x0) + (289L*x0_inner) + (221952L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x2) + (27744L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(0.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                            auto tmp7 = tmp5 - tmp6;
                            auto tmp8 = tmp4 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                            tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(110976L + x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (96L*x1) + (96L*x1_inner) + (27744L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (96L*x1) + (96L*x1_inner) + (27744L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                            tmp23.store(out_ptr3 + static_cast<long>(x2 + (96L*x1) + (96L*x1_inner) + (27744L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (96L*x1) + (27744L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(110976L + x1 + (289L*x2) + (221952L*x0))];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (96L*x1) + (27744L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp7 = out_ptr1[static_cast<long>(x2)];
                        auto tmp10 = in_ptr4[static_cast<long>(x2)];
                        auto tmp15 = out_ptr0[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = tmp0 ? tmp2 : tmp1;
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(0.00043252595155709344);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                        auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        auto tmp14 = decltype(tmp3)(tmp3 - tmp13);
                        auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                        auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                        auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                        auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                        out_ptr3[static_cast<long>(x2 + (96L*x1) + (27744L*x0))] = tmp20;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(288L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (289L*x0) + (221952L*x1)), static_cast<long>(289L), tmp1, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (289L*x0) + (221952L*x1)), static_cast<long>(289L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (110976L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (384L*x2_inner) + (110976L*x1)));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = static_cast<float>(0.0);
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                                auto tmp8 = tmp6 - tmp7;
                                auto tmp9 = tmp5 * tmp8;
                                tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                tmp_acc1_vec = tmp_acc1_vec + tmp9;
                            }
                        }
                        for(long x2=static_cast<long>(288L); x2<static_cast<long>(289L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (384L*x2) + (110976L*x1)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (289L*x0) + (289L*x0_inner) + (221952L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x2) + (110976L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(0.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                            auto tmp7 = tmp5 - tmp6;
                            auto tmp8 = tmp4 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                            tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (289L*x2) + (221952L*x0)), static_cast<long>(289L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (110976L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (110976L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = static_cast<float>(0.0);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = decltype(tmp4)::blendv(tmp2, tmp4, tmp0);
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(0.00043252595155709344);
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
                            tmp23.store(out_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (110976L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(288L); x1<static_cast<long>(289L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (110976L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (289L*x2) + (221952L*x0))];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (384L*x1) + (110976L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp7 = out_ptr1[static_cast<long>(x2)];
                        auto tmp10 = in_ptr4[static_cast<long>(x2)];
                        auto tmp15 = out_ptr0[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = tmp0 ? tmp2 : tmp1;
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(0.00043252595155709344);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                        auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        auto tmp14 = decltype(tmp3)(tmp3 - tmp13);
                        auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                        auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                        auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                        auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                        out_ptr3[static_cast<long>(x2 + (384L*x1) + (110976L*x0))] = tmp20;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(224L + x0 + (288L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(224L + x0 + (288L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(224L + x0 + (288L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(224L + x1 + (288L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(224L + x1 + (288L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(224L + x1 + (288L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00010204081632653062);
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
        #pragma omp single
        {
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
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_71 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(35L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(35L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (288L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(35L, 2L + x3))))) + (10080L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(35L, 2L + x2))))) + (352800L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (288L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (10080L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(35L, 2L + x2))))) + (352800L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (288L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (10080L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(35L, 2L + x2))))) + (352800L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (288L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(35L, 2L + x3))))) + (10080L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (352800L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(x1 + (288L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (10080L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (352800L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(x1 + (288L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (10080L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (352800L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(x1 + (288L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(35L, 2L + x3))))) + (10080L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (352800L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(x1 + (288L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (10080L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (352800L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(x1 + (288L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (10080L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (352800L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(35L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(35L, 2L + x3));
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
                            out_ptr0[static_cast<long>(x3 + (35L*x2) + (1225L*x1) + (352800L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(128L + x0 + (288L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x0 + (288L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(128L + x0 + (288L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(128L + x1 + (288L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x1 + (288L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(128L + x1 + (288L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00010204081632653062);
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
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x0 + (288L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(64L + x0 + (288L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(64L + x0 + (288L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (288L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(64L + x1 + (288L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(64L + x1 + (288L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00010204081632653062);
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
        #pragma omp single
        {
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (48L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (288L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (288L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (288L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (288L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (288L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = decltype(tmp7)::blendv(tmp5, tmp7, tmp0);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp13 = static_cast<float>(0.00010204081632653062);
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
        #pragma omp single
        {
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
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(274400L + x1 + (1225L*x2) + (352800L*x0)), static_cast<long>(1225L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(224L + x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(224L + x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(224L + x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(1224L); x1<static_cast<long>(1225L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(274400L + x1 + (1225L*x2) + (352800L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(224L + x2 + (288L*x1) + (352800L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(224L + x2 + (288L*x1) + (352800L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(224L + x2 + (288L*x1) + (352800L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))] = tmp9;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00010204081632653062);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(35L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(35L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(35L, 2L + x3))))) + (8960L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(35L, 2L + x2))))) + (313600L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (8960L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(35L, 2L + x2))))) + (313600L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (8960L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(35L, 2L + x2))))) + (313600L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(35L, 2L + x3))))) + (8960L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (313600L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (8960L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (313600L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (8960L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (313600L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(35L, 2L + x3))))) + (8960L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (313600L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (8960L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (313600L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(x1 + (256L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (8960L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (313600L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(35L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(35L, 2L + x3));
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
                            out_ptr0[static_cast<long>(x3 + (35L*x2) + (1225L*x1) + (313600L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(156800L + x1 + (1225L*x2) + (352800L*x0)), static_cast<long>(1225L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (96L*x1) + (96L*x1_inner) + (117600L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(128L + x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr1 + static_cast<long>(x2 + (96L*x1) + (96L*x1_inner) + (117600L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(1224L); x1<static_cast<long>(1225L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (96L*x1) + (117600L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(156800L + x1 + (1225L*x2) + (352800L*x0))];
                        auto tmp2 = in_ptr3[static_cast<long>(128L + x2 + (288L*x1) + (352800L*x0))];
                        auto tmp4 = in_ptr4[static_cast<long>(128L + x2 + (288L*x1) + (352800L*x0))];
                        auto tmp6 = in_ptr5[static_cast<long>(128L + x2 + (288L*x1) + (352800L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr1[static_cast<long>(x2 + (96L*x1) + (117600L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00010204081632653062);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(78400L + x1 + (1225L*x2) + (352800L*x0)), static_cast<long>(1225L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(64L + x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(64L + x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(64L + x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(1224L); x1<static_cast<long>(1225L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(78400L + x1 + (1225L*x2) + (352800L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(64L + x2 + (288L*x1) + (352800L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(64L + x2 + (288L*x1) + (352800L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(64L + x2 + (288L*x1) + (352800L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))] = tmp9;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00010204081632653062);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (48L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1225L*x2) + (352800L*x0)), static_cast<long>(1225L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (288L*x1) + (288L*x1_inner) + (352800L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(1224L); x1<static_cast<long>(1225L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1225L*x2) + (352800L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2 + (288L*x1) + (352800L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(x2 + (288L*x1) + (352800L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(x2 + (288L*x1) + (352800L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))] = tmp9;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00010204081632653062);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(274400L + x1 + (1225L*x2) + (313600L*x0)), static_cast<long>(1225L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (32L*x1) + (32L*x1_inner) + (39200L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(224L + x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(224L + x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(224L + x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (32L*x1_inner) + (39200L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(1224L); x1<static_cast<long>(1225L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (32L*x1) + (39200L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(274400L + x1 + (1225L*x2) + (313600L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(224L + x2 + (256L*x1) + (313600L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(224L + x2 + (256L*x1) + (313600L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(224L + x2 + (256L*x1) + (313600L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (32L*x1) + (39200L*x0))] = tmp9;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (32L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00010204081632653062);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(35L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(35L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (192L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(35L, 2L + x3))))) + (6720L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(35L, 2L + x2))))) + (235200L*x0))];
                            auto tmp11 = in_ptr0[static_cast<long>(x1 + (192L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (6720L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(35L, 2L + x2))))) + (235200L*x0))];
                            auto tmp18 = in_ptr0[static_cast<long>(x1 + (192L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (6720L*(std::min(std::max(0L, (-1L) + x2), (-1L) + (std::min(35L, 2L + x2))))) + (235200L*x0))];
                            auto tmp25 = in_ptr0[static_cast<long>(x1 + (192L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(35L, 2L + x3))))) + (6720L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (235200L*x0))];
                            auto tmp32 = in_ptr0[static_cast<long>(x1 + (192L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (6720L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (235200L*x0))];
                            auto tmp37 = in_ptr0[static_cast<long>(x1 + (192L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (6720L*(std::min(1L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (235200L*x0))];
                            auto tmp42 = in_ptr0[static_cast<long>(x1 + (192L*(std::min(std::max(0L, (-1L) + x3), (-1L) + (std::min(35L, 2L + x3))))) + (6720L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (235200L*x0))];
                            auto tmp49 = in_ptr0[static_cast<long>(x1 + (192L*(std::min(1L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (6720L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (235200L*x0))];
                            auto tmp54 = in_ptr0[static_cast<long>(x1 + (192L*(std::min(2L + (std::max(0L, (-1L) + x3)), (-1L) + (std::min(35L, 2L + x3))))) + (6720L*(std::min(2L + (std::max(0L, (-1L) + x2)), (-1L) + (std::min(35L, 2L + x2))))) + (235200L*x0))];
                            auto tmp1 = tmp0 / 9;
                            auto tmp2 = c10::convert<int>(std::max(0L, (-1L) + x2));
                            auto tmp3 = c10::convert<int>(std::min(35L, 2L + x2));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, (-1L) + x3));
                            auto tmp6 = c10::convert<int>(std::min(35L, 2L + x3));
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
                            out_ptr0[static_cast<long>(x3 + (35L*x2) + (1225L*x1) + (235200L*x0))] = tmp58;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(156800L + x1 + (1225L*x2) + (313600L*x0)), static_cast<long>(1225L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (96L*x1) + (96L*x1_inner) + (117600L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(128L + x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(128L + x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr1 + static_cast<long>(x2 + (96L*x1) + (96L*x1_inner) + (117600L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(1224L); x1<static_cast<long>(1225L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (96L*x1) + (117600L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(156800L + x1 + (1225L*x2) + (313600L*x0))];
                        auto tmp2 = in_ptr3[static_cast<long>(128L + x2 + (256L*x1) + (313600L*x0))];
                        auto tmp4 = in_ptr4[static_cast<long>(128L + x2 + (256L*x1) + (313600L*x0))];
                        auto tmp6 = in_ptr5[static_cast<long>(128L + x2 + (256L*x1) + (313600L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr1[static_cast<long>(x2 + (96L*x1) + (117600L*x0))] = tmp9;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00010204081632653062);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(78400L + x1 + (1225L*x2) + (313600L*x0)), static_cast<long>(1225L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(64L + x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(64L + x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(64L + x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(1224L); x1<static_cast<long>(1225L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(78400L + x1 + (1225L*x2) + (313600L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(64L + x2 + (256L*x1) + (313600L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(64L + x2 + (256L*x1) + (313600L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(64L + x2 + (256L*x1) + (313600L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))] = tmp9;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00010204081632653062);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (48L*x0)));
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
                    auto tmp10 = static_cast<float>(0.00010204081632653062);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1225L*x2) + (313600L*x0)), static_cast<long>(1225L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (313600L*x0)));
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            tmp11.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (78400L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(1224L); x1<static_cast<long>(1225L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1225L*x2) + (313600L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (313600L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(x2 + (256L*x1) + (313600L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(x2 + (256L*x1) + (313600L*x0))];
                        auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        out_ptr0[static_cast<long>(x2 + (64L*x1) + (78400L*x0))] = tmp9;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(9800L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00010204081632653062);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1224L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (1225L*x2) + (235200L*x0)), static_cast<long>(1225L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (235200L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (235200L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (235200L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            tmp7.store(in_out_ptr0 + static_cast<long>(x2 + (192L*x1) + (192L*x1_inner) + (235200L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(1224L); x1<static_cast<long>(1225L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1225L*x2) + (235200L*x0))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (192L*x1) + (235200L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (192L*x1) + (235200L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2 + (192L*x1) + (235200L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        in_out_ptr0[static_cast<long>(x2 + (192L*x1) + (235200L*x0))] = tmp6;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(40328L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40328L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp10 = static_cast<float>(2.479666732791113e-05);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(42632L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(42632L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
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
                    auto tmp10 = static_cast<float>(2.3456558453743668e-05);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(172872L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(172872L); x0+=static_cast<long>(1L))
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
                    auto tmp10 = static_cast<float>(5.78462677588042e-06);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(172872L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(172872L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    auto tmp10 = static_cast<float>(5.78462677588042e-06);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(177608L); x1+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(177608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
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
                    auto tmp10 = static_cast<float>(5.630377010044593e-06);
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_567, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, getitem_12, getitem_13, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, avg_pool2d, convolution_11, squeeze_34, cat, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, avg_pool2d_1, convolution_18, squeeze_55, cat_1, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, avg_pool2d_2, convolution_25, squeeze_76, cat_2, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, getitem_65, cat_3, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, avg_pool2d_3, convolution_39, squeeze_118, cat_4, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_41, convolution_42, squeeze_127, relu_42, convolution_43, squeeze_130, convolution_44, squeeze_133, relu_44, convolution_45, squeeze_136, relu_45, convolution_46, squeeze_139, relu_46, convolution_47, squeeze_142, relu_47, convolution_48, squeeze_145, avg_pool2d_4, convolution_49, squeeze_148, cat_5, convolution_50, squeeze_151, convolution_51, squeeze_154, relu_51, convolution_52, squeeze_157, relu_52, convolution_53, squeeze_160, convolution_54, squeeze_163, relu_54, convolution_55, squeeze_166, relu_55, convolution_56, squeeze_169, relu_56, convolution_57, squeeze_172, relu_57, convolution_58, squeeze_175, avg_pool2d_5, convolution_59, squeeze_178, cat_6, convolution_60, squeeze_181, convolution_61, squeeze_184, relu_61, convolution_62, squeeze_187, relu_62, convolution_63, squeeze_190, convolution_64, squeeze_193, relu_64, convolution_65, squeeze_196, relu_65, convolution_66, squeeze_199, relu_66, convolution_67, squeeze_202, relu_67, convolution_68, squeeze_205, avg_pool2d_6, convolution_69, squeeze_208, cat_7, convolution_70, squeeze_211, relu_70, convolution_71, squeeze_214, convolution_72, squeeze_217, relu_72, convolution_73, squeeze_220, relu_73, convolution_74, squeeze_223, relu_74, convolution_75, squeeze_226, getitem_159, cat_8, convolution_76, squeeze_229, convolution_77, squeeze_232, relu_77, convolution_78, squeeze_235, convolution_79, squeeze_238, convolution_80, squeeze_241, relu_80, convolution_81, squeeze_244, relu_81, convolution_82, squeeze_247, convolution_83, squeeze_250, avg_pool2d_7, convolution_84, squeeze_253, cat_11, convolution_85, squeeze_256, convolution_86, squeeze_259, relu_86, convolution_87, squeeze_262, convolution_88, squeeze_265, convolution_89, squeeze_268, relu_89, convolution_90, squeeze_271, relu_90, convolution_91, squeeze_274, convolution_92, squeeze_277, avg_pool2d_8, convolution_93, squeeze_280, clone, permute_1, le, unsqueeze_378, le_1, unsqueeze_390, le_2, unsqueeze_402, unsqueeze_414, unsqueeze_426, le_5, unsqueeze_438, le_6, unsqueeze_450, unsqueeze_462, le_8, unsqueeze_474, le_9, unsqueeze_486, le_10, unsqueeze_498, le_11, unsqueeze_510, unsqueeze_522, unsqueeze_534, le_14, unsqueeze_546, le_15, unsqueeze_558, unsqueeze_570, le_17, unsqueeze_582, le_18, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, le_22, unsqueeze_642, unsqueeze_654, le_24, unsqueeze_666, le_25, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, le_30, unsqueeze_738, unsqueeze_750, unsqueeze_762, le_33, unsqueeze_774, le_34, unsqueeze_786, le_35, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_40, unsqueeze_858, unsqueeze_870, unsqueeze_882, le_43, unsqueeze_894, le_44, unsqueeze_906, le_45, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, le_50, unsqueeze_978, unsqueeze_990, unsqueeze_1002, le_53, unsqueeze_1014, le_54, unsqueeze_1026, le_55, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, unsqueeze_1074, unsqueeze_1086, le_60, unsqueeze_1098, unsqueeze_1110, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, unsqueeze_1230, le_72, unsqueeze_1242, unsqueeze_1254, le_74, unsqueeze_1266, le_75, unsqueeze_1278, le_76, unsqueeze_1290, unsqueeze_1302, unsqueeze_1314, le_79, unsqueeze_1326, unsqueeze_1338, le_81, unsqueeze_1350, le_82, unsqueeze_1362, le_83, unsqueeze_1374, unsqueeze_1386, unsqueeze_1398, le_86, unsqueeze_1410, unsqueeze_1422, le_88, unsqueeze_1434, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1482, unsqueeze_1494, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (80, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (48, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_27, (48, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_39, (64, ), (1, ))
    assert_size_stride(primals_41, (48, ), (1, ))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_47, (96, ), (1, ))
    assert_size_stride(primals_49, (96, ), (1, ))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_53, (384, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_57, (96, ), (1, ))
    assert_size_stride(primals_59, (96, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_67, (192, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_75, (128, ), (1, ))
    assert_size_stride(primals_77, (192, ), (1, ))
    assert_size_stride(primals_79, (192, ), (1, ))
    assert_size_stride(primals_81, (192, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_85, (160, ), (1, ))
    assert_size_stride(primals_87, (192, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_91, (160, ), (1, ))
    assert_size_stride(primals_93, (160, ), (1, ))
    assert_size_stride(primals_95, (160, ), (1, ))
    assert_size_stride(primals_97, (192, ), (1, ))
    assert_size_stride(primals_99, (192, ), (1, ))
    assert_size_stride(primals_101, (192, ), (1, ))
    assert_size_stride(primals_103, (160, ), (1, ))
    assert_size_stride(primals_105, (160, ), (1, ))
    assert_size_stride(primals_107, (192, ), (1, ))
    assert_size_stride(primals_109, (160, ), (1, ))
    assert_size_stride(primals_111, (160, ), (1, ))
    assert_size_stride(primals_113, (160, ), (1, ))
    assert_size_stride(primals_115, (160, ), (1, ))
    assert_size_stride(primals_117, (192, ), (1, ))
    assert_size_stride(primals_119, (192, ), (1, ))
    assert_size_stride(primals_121, (192, ), (1, ))
    assert_size_stride(primals_123, (192, ), (1, ))
    assert_size_stride(primals_125, (192, ), (1, ))
    assert_size_stride(primals_127, (192, ), (1, ))
    assert_size_stride(primals_129, (192, ), (1, ))
    assert_size_stride(primals_131, (192, ), (1, ))
    assert_size_stride(primals_133, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_137, (192, ), (1, ))
    assert_size_stride(primals_139, (192, ), (1, ))
    assert_size_stride(primals_141, (192, ), (1, ))
    assert_size_stride(primals_143, (320, ), (1, ))
    assert_size_stride(primals_145, (192, ), (1, ))
    assert_size_stride(primals_147, (192, ), (1, ))
    assert_size_stride(primals_149, (192, ), (1, ))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_153, (320, ), (1, ))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_161, (448, ), (1, ))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_169, (192, ), (1, ))
    assert_size_stride(primals_171, (320, ), (1, ))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_179, (448, ), (1, ))
    assert_size_stride(primals_181, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_185, (384, ), (1, ))
    assert_size_stride(primals_187, (192, ), (1, ))
    assert_size_stride(primals_189, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_190, (32, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_191, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_192, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_193, (192, 80, 3, 3), (720, 1, 240, 80))
    assert_size_stride(primals_194, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_195, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_196, (64, 48, 5, 5), (1200, 1, 240, 48))
    assert_size_stride(primals_197, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_198, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_199, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_200, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_201, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_202, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_203, (64, 48, 5, 5), (1200, 1, 240, 48))
    assert_size_stride(primals_204, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_205, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_206, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_207, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_208, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_209, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_210, (64, 48, 5, 5), (1200, 1, 240, 48))
    assert_size_stride(primals_211, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_212, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_213, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_214, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_215, (384, 288, 3, 3), (2592, 1, 864, 288))
    assert_size_stride(primals_216, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_217, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_218, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_219, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_220, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_221, (128, 128, 1, 7), (896, 1, 896, 128))
    assert_size_stride(primals_222, (192, 128, 7, 1), (896, 1, 128, 128))
    assert_size_stride(primals_223, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_224, (128, 128, 7, 1), (896, 1, 128, 128))
    assert_size_stride(primals_225, (128, 128, 1, 7), (896, 1, 896, 128))
    assert_size_stride(primals_226, (128, 128, 7, 1), (896, 1, 128, 128))
    assert_size_stride(primals_227, (192, 128, 1, 7), (896, 1, 896, 128))
    assert_size_stride(primals_228, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_229, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_230, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_231, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_232, (192, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_233, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_234, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_235, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_236, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_237, (192, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_238, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_239, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_240, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_241, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_242, (192, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_243, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_244, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_245, (160, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_246, (160, 160, 7, 1), (1120, 1, 160, 160))
    assert_size_stride(primals_247, (192, 160, 1, 7), (1120, 1, 1120, 160))
    assert_size_stride(primals_248, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_249, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_250, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_251, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_252, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_253, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_254, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_255, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_256, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_257, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_258, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_259, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_260, (320, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_261, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_262, (192, 192, 1, 7), (1344, 1, 1344, 192))
    assert_size_stride(primals_263, (192, 192, 7, 1), (1344, 1, 192, 192))
    assert_size_stride(primals_264, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_265, (320, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_266, (384, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_267, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_268, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_269, (448, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_270, (384, 448, 3, 3), (4032, 1, 1344, 448))
    assert_size_stride(primals_271, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_272, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_273, (192, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_274, (320, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_275, (384, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_276, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_277, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_278, (448, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_279, (384, 448, 3, 3), (4032, 1, 1344, 448))
    assert_size_stride(primals_280, (384, 384, 1, 3), (1152, 1, 1152, 384))
    assert_size_stride(primals_281, (384, 384, 3, 1), (1152, 1, 384, 384))
    assert_size_stride(primals_282, (192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_567, (8, 3, 299, 299), (268203, 1, 897, 3))
    assert_size_stride(convolution, (8, 32, 149, 149), (710432, 1, 4768, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 149, 149), (710432, 1, 4768, 32))
    assert_size_stride(convolution_1, (8, 32, 147, 147), (691488, 1, 4704, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(relu_1, (8, 32, 147, 147), (691488, 1, 4704, 32))
    assert_size_stride(convolution_2, (8, 64, 147, 147), (1382976, 1, 9408, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (8, 64, 147, 147), (1382976, 1, 9408, 64))
    assert_size_stride(getitem_6, (8, 64, 73, 73), (341056, 1, 4672, 64))
    assert_size_stride(getitem_7, (8, 64, 73, 73), (341056, 1, 4672, 64))
    assert_size_stride(convolution_3, (8, 80, 73, 73), (426320, 1, 5840, 80))
    assert_size_stride(squeeze_10, (80, ), (1, ))
    assert_size_stride(relu_3, (8, 80, 73, 73), (426320, 1, 5840, 80))
    assert_size_stride(convolution_4, (8, 192, 71, 71), (967872, 1, 13632, 192))
    assert_size_stride(squeeze_13, (192, ), (1, ))
    assert_size_stride(relu_4, (8, 192, 71, 71), (967872, 1, 13632, 192))
    assert_size_stride(getitem_12, (8, 192, 35, 35), (235200, 1, 6720, 192))
    assert_size_stride(getitem_13, (8, 192, 35, 35), (235200, 1, 6720, 192))
    assert_size_stride(convolution_5, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(convolution_6, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(squeeze_19, (48, ), (1, ))
    assert_size_stride(relu_6, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(convolution_7, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(convolution_8, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_8, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_9, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_28, (96, ), (1, ))
    assert_size_stride(relu_9, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_10, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_31, (96, ), (1, ))
    assert_size_stride(avg_pool2d, (8, 192, 35, 35), (235200, 1, 6720, 192))
    assert_size_stride(convolution_11, (8, 32, 35, 35), (39200, 1, 1120, 32))
    assert_size_stride(squeeze_34, (32, ), (1, ))
    assert_size_stride(cat, (8, 256, 35, 35), (313600, 1, 8960, 256))
    assert_size_stride(convolution_12, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_37, (64, ), (1, ))
    assert_size_stride(convolution_13, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(squeeze_40, (48, ), (1, ))
    assert_size_stride(relu_13, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(convolution_14, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_43, (64, ), (1, ))
    assert_size_stride(convolution_15, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_46, (64, ), (1, ))
    assert_size_stride(relu_15, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_16, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_49, (96, ), (1, ))
    assert_size_stride(relu_16, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_17, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_52, (96, ), (1, ))
    assert_size_stride(avg_pool2d_1, (8, 256, 35, 35), (313600, 1, 8960, 256))
    assert_size_stride(convolution_18, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_55, (64, ), (1, ))
    assert_size_stride(cat_1, (8, 288, 35, 35), (352800, 1, 10080, 288))
    assert_size_stride(convolution_19, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_58, (64, ), (1, ))
    assert_size_stride(convolution_20, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(squeeze_61, (48, ), (1, ))
    assert_size_stride(relu_20, (8, 48, 35, 35), (58800, 1, 1680, 48))
    assert_size_stride(convolution_21, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_64, (64, ), (1, ))
    assert_size_stride(convolution_22, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_67, (64, ), (1, ))
    assert_size_stride(relu_22, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_23, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_70, (96, ), (1, ))
    assert_size_stride(relu_23, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_24, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_73, (96, ), (1, ))
    assert_size_stride(avg_pool2d_2, (8, 288, 35, 35), (352800, 1, 10080, 288))
    assert_size_stride(convolution_25, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_76, (64, ), (1, ))
    assert_size_stride(cat_2, (8, 288, 35, 35), (352800, 1, 10080, 288))
    assert_size_stride(convolution_26, (8, 384, 17, 17), (110976, 1, 6528, 384))
    assert_size_stride(squeeze_79, (384, ), (1, ))
    assert_size_stride(convolution_27, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(squeeze_82, (64, ), (1, ))
    assert_size_stride(relu_27, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(convolution_28, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(squeeze_85, (96, ), (1, ))
    assert_size_stride(relu_28, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(convolution_29, (8, 96, 17, 17), (27744, 1, 1632, 96))
    assert_size_stride(squeeze_88, (96, ), (1, ))
    assert_size_stride(getitem_65, (8, 288, 17, 17), (83232, 1, 4896, 288))
    assert_size_stride(cat_3, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_30, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_91, (192, ), (1, ))
    assert_size_stride(convolution_31, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_94, (128, ), (1, ))
    assert_size_stride(relu_31, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_32, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_97, (128, ), (1, ))
    assert_size_stride(relu_32, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_33, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_100, (192, ), (1, ))
    assert_size_stride(convolution_34, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_103, (128, ), (1, ))
    assert_size_stride(relu_34, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_35, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_106, (128, ), (1, ))
    assert_size_stride(relu_35, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_36, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_109, (128, ), (1, ))
    assert_size_stride(relu_36, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_37, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(squeeze_112, (128, ), (1, ))
    assert_size_stride(relu_37, (8, 128, 17, 17), (36992, 1, 2176, 128))
    assert_size_stride(convolution_38, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_115, (192, ), (1, ))
    assert_size_stride(avg_pool2d_3, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_39, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_118, (192, ), (1, ))
    assert_size_stride(cat_4, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_40, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_121, (192, ), (1, ))
    assert_size_stride(convolution_41, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_124, (160, ), (1, ))
    assert_size_stride(relu_41, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_42, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_127, (160, ), (1, ))
    assert_size_stride(relu_42, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_43, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_130, (192, ), (1, ))
    assert_size_stride(convolution_44, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_133, (160, ), (1, ))
    assert_size_stride(relu_44, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_45, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_136, (160, ), (1, ))
    assert_size_stride(relu_45, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_46, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_139, (160, ), (1, ))
    assert_size_stride(relu_46, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_47, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_142, (160, ), (1, ))
    assert_size_stride(relu_47, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_48, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_145, (192, ), (1, ))
    assert_size_stride(avg_pool2d_4, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_49, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_148, (192, ), (1, ))
    assert_size_stride(cat_5, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_50, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_151, (192, ), (1, ))
    assert_size_stride(convolution_51, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_154, (160, ), (1, ))
    assert_size_stride(relu_51, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_52, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_157, (160, ), (1, ))
    assert_size_stride(relu_52, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_53, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_160, (192, ), (1, ))
    assert_size_stride(convolution_54, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_163, (160, ), (1, ))
    assert_size_stride(relu_54, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_55, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_166, (160, ), (1, ))
    assert_size_stride(relu_55, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_56, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_169, (160, ), (1, ))
    assert_size_stride(relu_56, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_57, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(squeeze_172, (160, ), (1, ))
    assert_size_stride(relu_57, (8, 160, 17, 17), (46240, 1, 2720, 160))
    assert_size_stride(convolution_58, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_175, (192, ), (1, ))
    assert_size_stride(avg_pool2d_5, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_59, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_178, (192, ), (1, ))
    assert_size_stride(cat_6, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_60, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_181, (192, ), (1, ))
    assert_size_stride(convolution_61, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_184, (192, ), (1, ))
    assert_size_stride(relu_61, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_62, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_187, (192, ), (1, ))
    assert_size_stride(relu_62, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_63, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_190, (192, ), (1, ))
    assert_size_stride(convolution_64, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_193, (192, ), (1, ))
    assert_size_stride(relu_64, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_65, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_196, (192, ), (1, ))
    assert_size_stride(relu_65, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_66, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_199, (192, ), (1, ))
    assert_size_stride(relu_66, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_67, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_202, (192, ), (1, ))
    assert_size_stride(relu_67, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_68, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_205, (192, ), (1, ))
    assert_size_stride(avg_pool2d_6, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_69, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_208, (192, ), (1, ))
    assert_size_stride(cat_7, (8, 768, 17, 17), (221952, 1, 13056, 768))
    assert_size_stride(convolution_70, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_211, (192, ), (1, ))
    assert_size_stride(relu_70, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_71, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(squeeze_214, (320, ), (1, ))
    assert_size_stride(convolution_72, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_217, (192, ), (1, ))
    assert_size_stride(relu_72, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_73, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_220, (192, ), (1, ))
    assert_size_stride(relu_73, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_74, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(squeeze_223, (192, ), (1, ))
    assert_size_stride(relu_74, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(convolution_75, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(squeeze_226, (192, ), (1, ))
    assert_size_stride(getitem_159, (8, 768, 8, 8), (49152, 1, 6144, 768))
    assert_size_stride(cat_8, (8, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(convolution_76, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(squeeze_229, (320, ), (1, ))
    assert_size_stride(convolution_77, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_232, (384, ), (1, ))
    assert_size_stride(relu_77, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_78, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_235, (384, ), (1, ))
    assert_size_stride(convolution_79, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_238, (384, ), (1, ))
    assert_size_stride(convolution_80, (8, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(squeeze_241, (448, ), (1, ))
    assert_size_stride(relu_80, (8, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(convolution_81, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_244, (384, ), (1, ))
    assert_size_stride(relu_81, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_82, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_247, (384, ), (1, ))
    assert_size_stride(convolution_83, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_250, (384, ), (1, ))
    assert_size_stride(avg_pool2d_7, (8, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(convolution_84, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(squeeze_253, (192, ), (1, ))
    assert_size_stride(cat_11, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(convolution_85, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(squeeze_256, (320, ), (1, ))
    assert_size_stride(convolution_86, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_259, (384, ), (1, ))
    assert_size_stride(relu_86, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_87, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_262, (384, ), (1, ))
    assert_size_stride(convolution_88, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_265, (384, ), (1, ))
    assert_size_stride(convolution_89, (8, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(squeeze_268, (448, ), (1, ))
    assert_size_stride(relu_89, (8, 448, 8, 8), (28672, 1, 3584, 448))
    assert_size_stride(convolution_90, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_271, (384, ), (1, ))
    assert_size_stride(relu_90, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(convolution_91, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_274, (384, ), (1, ))
    assert_size_stride(convolution_92, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(squeeze_277, (384, ), (1, ))
    assert_size_stride(avg_pool2d_8, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(convolution_93, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(squeeze_280, (192, ), (1, ))
    assert_size_stride(clone, (8, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(unsqueeze_378, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_1, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_390, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_2, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_402, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(le_5, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_438, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_6, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_450, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_8, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(unsqueeze_474, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(le_9, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(unsqueeze_486, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_10, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_498, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_11, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_510, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_522, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(le_14, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_546, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_15, (8, 384, 8, 8), (24576, 1, 3072, 384))
    assert_size_stride(unsqueeze_558, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_17, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(unsqueeze_582, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(le_18, (8, 192, 8, 8), (12288, 1, 1536, 192))
    assert_size_stride(unsqueeze_594, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_630, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_22, (8, 320, 8, 8), (20480, 1, 2560, 320))
    assert_size_stride(unsqueeze_642, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_24, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_666, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_25, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_678, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_702, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_30, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_738, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_33, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_774, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_34, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_786, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_35, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_798, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_810, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_822, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_846, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_40, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_858, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_870, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_882, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_43, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_894, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_44, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_906, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_45, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_918, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_930, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_942, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_954, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_966, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_50, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_978, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_990, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_1002, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(le_53, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1014, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_54, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1026, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_55, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1038, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1050, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1062, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1074, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1086, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_60, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1098, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1110, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_1122, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(le_63, (8, 192, 17, 17), (55488, 1, 3264, 192))
    assert_size_stride(unsqueeze_1134, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_64, (8, 96, 17, 17), (27744, 1, 1632, 96))
    assert_size_stride(unsqueeze_1146, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1158, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1170, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_67, (8, 384, 17, 17), (110976, 1, 6528, 384))
    assert_size_stride(unsqueeze_1182, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(le_68, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1194, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_69, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(unsqueeze_1206, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1218, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1230, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_72, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1242, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1254, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_74, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1266, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_75, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1278, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_76, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(unsqueeze_1290, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1302, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1314, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_79, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1326, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1338, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_81, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1350, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_82, (8, 32, 35, 35), (39200, 1, 1120, 32))
    assert_size_stride(unsqueeze_1362, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(le_83, (8, 96, 35, 35), (117600, 1, 3360, 96))
    assert_size_stride(unsqueeze_1374, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1386, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_1398, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(le_86, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1410, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1422, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_88, (8, 64, 35, 35), (78400, 1, 2240, 64))
    assert_size_stride(unsqueeze_1434, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1446, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_1458, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_1470, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_1482, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_1494, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
    del clone
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((192, ), device='cpu', dtype=torch.float32)
    buf4 = empty((192, ), device='cpu', dtype=torch.float32)
    buf5 = empty((192, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_sum_threshold_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_93.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_280.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del convolution_93
    del le
    del primals_187
    del squeeze_280
    del tangents_1
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, avg_pool2d_8, primals_282, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_8
    del primals_282
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((8, 2048, 8, 8), device='cpu', dtype=torch.float32)
    buf11 = empty((384, ), device='cpu', dtype=torch.float32)
    buf12 = empty((384, ), device='cpu', dtype=torch.float32)
    buf13 = empty((384, ), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_1(c_void_p(buf8.data_ptr()), c_void_p(le_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_92.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_277.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()))
    del buf8
    del convolution_92
    del le_1
    del primals_185
    del squeeze_277
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf15 = aten.convolution_backward(buf14, relu_90, primals_281, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_281
    buf16 = buf15[0]
    buf17 = buf15[1]
    del buf15
    buf18 = buf12; del buf12  # reuse
    buf19 = empty((384, ), device='cpu', dtype=torch.float32)
    buf20 = empty((384, ), device='cpu', dtype=torch.float32)
    buf21 = buf14; del buf14  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2(c_void_p(le_2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_91.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_274.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del convolution_91
    del le_2
    del primals_183
    del squeeze_274
    del unsqueeze_402
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf22 = aten.convolution_backward(buf21, relu_90, primals_280, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf21
    del primals_280
    buf23 = buf22[0]
    buf24 = buf22[1]
    del buf22
    buf25 = buf19; del buf19  # reuse
    buf26 = empty((384, ), device='cpu', dtype=torch.float32)
    buf27 = buf16; del buf16  # reuse
    buf28 = buf26; del buf26  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_3(c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(relu_90.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(convolution_90.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(squeeze_271.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(buf25.data_ptr()))
    del buf23
    del convolution_90
    del primals_181
    del relu_90
    del squeeze_271
    del unsqueeze_414
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf29 = aten.convolution_backward(buf27, relu_89, primals_279, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_279
    buf30 = buf29[0]
    buf31 = buf29[1]
    del buf29
    buf32 = empty((448, ), device='cpu', dtype=torch.float32)
    buf33 = empty((448, ), device='cpu', dtype=torch.float32)
    buf34 = empty((448, ), device='cpu', dtype=torch.float32)
    buf35 = buf30; del buf30  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf35.data_ptr()), c_void_p(relu_89.data_ptr()), c_void_p(convolution_89.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_268.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del convolution_89
    del primals_179
    del relu_89
    del squeeze_268
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf36 = aten.convolution_backward(buf35, cat_11, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf35
    del primals_278
    buf37 = buf36[0]
    buf38 = buf36[1]
    del buf36
    buf39 = empty((384, ), device='cpu', dtype=torch.float32)
    buf40 = empty((384, ), device='cpu', dtype=torch.float32)
    buf41 = empty((384, ), device='cpu', dtype=torch.float32)
    buf42 = buf27; del buf27  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(le_5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_88.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_265.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()))
    del convolution_88
    del le_5
    del primals_177
    del squeeze_265
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf43 = aten.convolution_backward(buf42, relu_86, primals_277, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_277
    buf44 = buf43[0]
    buf45 = buf43[1]
    del buf43
    buf46 = buf40; del buf40  # reuse
    buf47 = empty((384, ), device='cpu', dtype=torch.float32)
    buf48 = empty((384, ), device='cpu', dtype=torch.float32)
    buf49 = buf42; del buf42  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(le_6.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_87.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_262.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    del convolution_87
    del le_6
    del primals_175
    del squeeze_262
    del unsqueeze_450
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf50 = aten.convolution_backward(buf49, relu_86, primals_276, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf49
    del primals_276
    buf51 = buf50[0]
    buf52 = buf50[1]
    del buf50
    buf53 = buf47; del buf47  # reuse
    buf54 = empty((384, ), device='cpu', dtype=torch.float32)
    buf55 = buf44; del buf44  # reuse
    buf56 = buf54; del buf54  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_7(c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(relu_86.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(convolution_86.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(squeeze_259.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf53.data_ptr()))
    del buf51
    del convolution_86
    del primals_173
    del relu_86
    del squeeze_259
    del unsqueeze_462
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf57 = aten.convolution_backward(buf55, cat_11, primals_275, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_275
    buf58 = buf57[0]
    buf59 = buf57[1]
    del buf57
    buf60 = empty((320, ), device='cpu', dtype=torch.float32)
    buf61 = empty((320, ), device='cpu', dtype=torch.float32)
    buf62 = empty((320, ), device='cpu', dtype=torch.float32)
    buf63 = empty_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8(c_void_p(le_8.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_85.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_256.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    del buf0
    del convolution_85
    del le_8
    del primals_171
    del squeeze_256
    del unsqueeze_474
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf64 = aten.convolution_backward(buf63, cat_11, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_11
    del primals_274
    buf65 = buf64[0]
    buf66 = buf64[1]
    del buf64
    buf67 = buf6; del buf6  # reuse
    buf68 = buf4; del buf4  # reuse
    buf69 = empty((192, ), device='cpu', dtype=torch.float32)
    buf70 = empty((192, ), device='cpu', dtype=torch.float32)
    buf71 = buf67; del buf67  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf71.data_ptr()), c_void_p(le_9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(convolution_84.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(squeeze_253.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del convolution_84
    del le_9
    del primals_169
    del squeeze_253
    del unsqueeze_486
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf72 = aten.convolution_backward(buf71, avg_pool2d_7, primals_273, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_7
    del primals_273
    buf73 = buf72[0]
    buf74 = buf72[1]
    del buf72
    buf75 = empty((8, 1280, 8, 8), device='cpu', dtype=torch.float32)
    buf76 = buf55; del buf55  # reuse
    buf77 = empty((384, ), device='cpu', dtype=torch.float32)
    buf78 = empty((384, ), device='cpu', dtype=torch.float32)
    buf79 = empty((384, ), device='cpu', dtype=torch.float32)
    buf80 = buf76; del buf76  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf80.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(le_10.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(convolution_83.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_250.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del buf73
    del convolution_83
    del le_10
    del primals_167
    del squeeze_250
    del unsqueeze_498
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf81 = aten.convolution_backward(buf80, relu_81, primals_272, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_272
    buf82 = buf81[0]
    buf83 = buf81[1]
    del buf81
    buf84 = buf80; del buf80  # reuse
    buf85 = buf78; del buf78  # reuse
    buf86 = empty((384, ), device='cpu', dtype=torch.float32)
    buf87 = empty((384, ), device='cpu', dtype=torch.float32)
    buf88 = buf84; del buf84  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(buf88.data_ptr()), c_void_p(le_11.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(convolution_82.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(squeeze_247.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    del convolution_82
    del le_11
    del primals_165
    del squeeze_247
    del unsqueeze_510
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf89 = aten.convolution_backward(buf88, relu_81, primals_271, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf88
    del primals_271
    buf90 = buf89[0]
    buf91 = buf89[1]
    del buf89
    buf92 = buf86; del buf86  # reuse
    buf93 = empty((384, ), device='cpu', dtype=torch.float32)
    buf94 = buf82; del buf82  # reuse
    buf95 = buf93; del buf93  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_12(c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(relu_81.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(convolution_81.data_ptr()), c_void_p(unsqueeze_522.data_ptr()), c_void_p(squeeze_244.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf92.data_ptr()))
    del buf90
    del convolution_81
    del primals_163
    del relu_81
    del squeeze_244
    del unsqueeze_522
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf96 = aten.convolution_backward(buf94, relu_80, primals_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_270
    buf97 = buf96[0]
    buf98 = buf96[1]
    del buf96
    buf99 = buf33; del buf33  # reuse
    buf100 = empty((448, ), device='cpu', dtype=torch.float32)
    buf101 = empty((448, ), device='cpu', dtype=torch.float32)
    buf102 = buf97; del buf97  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(buf102.data_ptr()), c_void_p(relu_80.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(unsqueeze_534.data_ptr()), c_void_p(squeeze_241.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del buf100
    del convolution_80
    del primals_161
    del relu_80
    del squeeze_241
    del unsqueeze_534
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf103 = aten.convolution_backward(buf102, cat_8, primals_269, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf102
    del primals_269
    buf104 = buf103[0]
    buf105 = buf103[1]
    del buf103
    buf106 = buf94; del buf94  # reuse
    buf107 = empty((384, ), device='cpu', dtype=torch.float32)
    buf108 = empty((384, ), device='cpu', dtype=torch.float32)
    buf109 = empty((384, ), device='cpu', dtype=torch.float32)
    buf110 = buf106; del buf106  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf110.data_ptr()), c_void_p(le_14.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(squeeze_238.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del convolution_79
    del le_14
    del primals_159
    del squeeze_238
    del unsqueeze_546
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf111 = aten.convolution_backward(buf110, relu_77, primals_268, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_268
    buf112 = buf111[0]
    buf113 = buf111[1]
    del buf111
    buf114 = buf110; del buf110  # reuse
    buf115 = buf108; del buf108  # reuse
    buf116 = empty((384, ), device='cpu', dtype=torch.float32)
    buf117 = empty((384, ), device='cpu', dtype=torch.float32)
    buf118 = buf114; del buf114  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf118.data_ptr()), c_void_p(le_15.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(convolution_78.data_ptr()), c_void_p(unsqueeze_558.data_ptr()), c_void_p(squeeze_235.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del convolution_78
    del le_15
    del primals_157
    del squeeze_235
    del unsqueeze_558
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf119 = aten.convolution_backward(buf118, relu_77, primals_267, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf118
    del primals_267
    buf120 = buf119[0]
    buf121 = buf119[1]
    del buf119
    buf122 = buf116; del buf116  # reuse
    buf123 = empty((384, ), device='cpu', dtype=torch.float32)
    buf124 = buf112; del buf112  # reuse
    buf125 = buf123; del buf123  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_16(c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(relu_77.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(unsqueeze_570.data_ptr()), c_void_p(squeeze_232.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf122.data_ptr()))
    del buf120
    del convolution_77
    del primals_155
    del relu_77
    del squeeze_232
    del unsqueeze_570
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf126 = aten.convolution_backward(buf124, cat_8, primals_266, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf124
    del primals_266
    buf127 = buf126[0]
    buf128 = buf126[1]
    del buf126
    buf129 = buf63; del buf63  # reuse
    buf130 = buf61; del buf61  # reuse
    buf131 = empty((320, ), device='cpu', dtype=torch.float32)
    buf132 = empty((320, ), device='cpu', dtype=torch.float32)
    buf133 = buf129; del buf129  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(buf133.data_ptr()), c_void_p(le_17.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(squeeze_229.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del buf10
    del buf37
    del buf58
    del buf65
    del convolution_76
    del le_17
    del primals_153
    del squeeze_229
    del unsqueeze_582
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf134 = aten.convolution_backward(buf133, cat_8, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_8
    del primals_265
    buf135 = buf134[0]
    buf136 = buf134[1]
    del buf134
    buf137 = buf75; del buf75  # reuse
    cpp_fused_add_18(c_void_p(buf137.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf135.data_ptr()))
    del buf104
    del buf127
    del buf135
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf138 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf137, (8, 768, 8, 8), (81920, 64, 8, 1), 32768), cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_159)
    del getitem_159
    buf139 = buf138
    del buf138
    buf140 = buf69; del buf69  # reuse
    buf141 = empty((192, ), device='cpu', dtype=torch.float32)
    buf142 = empty((192, ), device='cpu', dtype=torch.float32)
    buf143 = buf71; del buf71  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(le_18.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(unsqueeze_594.data_ptr()), c_void_p(squeeze_226.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    del convolution_75
    del le_18
    del primals_151
    del squeeze_226
    del unsqueeze_594
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf144 = aten.convolution_backward(buf143, relu_74, primals_264, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf143
    del primals_264
    buf145 = buf144[0]
    buf146 = buf144[1]
    del buf144
    buf147 = buf141; del buf141  # reuse
    buf148 = empty((192, ), device='cpu', dtype=torch.float32)
    buf149 = empty((192, ), device='cpu', dtype=torch.float32)
    buf150 = buf145; del buf145  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(buf150.data_ptr()), c_void_p(relu_74.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(unsqueeze_606.data_ptr()), c_void_p(squeeze_223.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    del convolution_74
    del primals_149
    del relu_74
    del squeeze_223
    del unsqueeze_606
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf151 = aten.convolution_backward(buf150, relu_73, primals_263, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf150
    del primals_263
    buf152 = buf151[0]
    buf153 = buf151[1]
    del buf151
    buf154 = buf148; del buf148  # reuse
    buf155 = empty((192, ), device='cpu', dtype=torch.float32)
    buf156 = empty((192, ), device='cpu', dtype=torch.float32)
    buf157 = buf152; del buf152  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21(c_void_p(buf157.data_ptr()), c_void_p(relu_73.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(squeeze_220.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    del convolution_73
    del primals_147
    del relu_73
    del squeeze_220
    del unsqueeze_618
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf158 = aten.convolution_backward(buf157, relu_72, primals_262, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf157
    del primals_262
    buf159 = buf158[0]
    buf160 = buf158[1]
    del buf158
    buf161 = buf155; del buf155  # reuse
    buf162 = empty((192, ), device='cpu', dtype=torch.float32)
    buf163 = empty((192, ), device='cpu', dtype=torch.float32)
    buf164 = buf159; del buf159  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_22(c_void_p(buf164.data_ptr()), c_void_p(relu_72.data_ptr()), c_void_p(convolution_72.data_ptr()), c_void_p(unsqueeze_630.data_ptr()), c_void_p(squeeze_217.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    del convolution_72
    del primals_145
    del relu_72
    del squeeze_217
    del unsqueeze_630
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf165 = aten.convolution_backward(buf164, cat_7, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf164
    del primals_261
    buf166 = buf165[0]
    buf167 = buf165[1]
    del buf165
    buf168 = buf131; del buf131  # reuse
    buf169 = empty((320, ), device='cpu', dtype=torch.float32)
    buf170 = empty((320, ), device='cpu', dtype=torch.float32)
    buf171 = buf133; del buf133  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(le_22.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(unsqueeze_642.data_ptr()), c_void_p(squeeze_214.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    del buf137
    del buf169
    del convolution_71
    del le_22
    del primals_143
    del squeeze_214
    del unsqueeze_642
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf172 = aten.convolution_backward(buf171, relu_70, primals_260, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf171
    del primals_260
    buf173 = buf172[0]
    buf174 = buf172[1]
    del buf172
    buf175 = buf162; del buf162  # reuse
    buf176 = empty((192, ), device='cpu', dtype=torch.float32)
    buf177 = empty((192, ), device='cpu', dtype=torch.float32)
    buf178 = buf173; del buf173  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf178.data_ptr()), c_void_p(relu_70.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(squeeze_211.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    del convolution_70
    del primals_141
    del relu_70
    del squeeze_211
    del unsqueeze_654
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf179 = aten.convolution_backward(buf178, cat_7, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_7
    del primals_259
    buf180 = buf179[0]
    buf181 = buf179[1]
    del buf179
    buf182 = buf176; del buf176  # reuse
    buf183 = empty((192, ), device='cpu', dtype=torch.float32)
    buf184 = buf178; del buf178  # reuse
    buf186 = buf184; del buf184  # reuse
    buf185 = buf183; del buf183  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_25(c_void_p(buf186.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(le_24.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(unsqueeze_666.data_ptr()), c_void_p(squeeze_208.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(buf182.data_ptr()))
    del convolution_69
    del le_24
    del primals_139
    del squeeze_208
    del unsqueeze_666
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf187 = aten.convolution_backward(buf186, avg_pool2d_6, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_6
    del primals_258
    buf188 = buf187[0]
    buf189 = buf187[1]
    del buf187
    buf190 = empty((8, 768, 17, 17), device='cpu', dtype=torch.float32)
    buf191 = empty((192, ), device='cpu', dtype=torch.float32)
    buf192 = empty((192, ), device='cpu', dtype=torch.float32)
    buf193 = buf186; del buf186  # reuse
    buf195 = buf193; del buf193  # reuse
    buf194 = buf192; del buf192  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_26(c_void_p(buf195.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(le_25.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(unsqueeze_678.data_ptr()), c_void_p(squeeze_205.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    del buf188
    del convolution_68
    del le_25
    del primals_137
    del squeeze_205
    del unsqueeze_678
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf196 = aten.convolution_backward(buf195, relu_67, primals_257, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf195
    del primals_257
    buf197 = buf196[0]
    buf198 = buf196[1]
    del buf196
    buf199 = empty((192, ), device='cpu', dtype=torch.float32)
    buf200 = empty((192, ), device='cpu', dtype=torch.float32)
    buf201 = empty((192, ), device='cpu', dtype=torch.float32)
    buf202 = buf197; del buf197  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(buf202.data_ptr()), c_void_p(relu_67.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(squeeze_202.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    del convolution_67
    del primals_135
    del relu_67
    del squeeze_202
    del unsqueeze_690
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf203 = aten.convolution_backward(buf202, relu_66, primals_256, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf202
    del primals_256
    buf204 = buf203[0]
    buf205 = buf203[1]
    del buf203
    buf206 = buf200; del buf200  # reuse
    buf207 = empty((192, ), device='cpu', dtype=torch.float32)
    buf208 = empty((192, ), device='cpu', dtype=torch.float32)
    buf209 = buf204; del buf204  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28(c_void_p(buf209.data_ptr()), c_void_p(relu_66.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(unsqueeze_702.data_ptr()), c_void_p(squeeze_199.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del convolution_66
    del primals_133
    del relu_66
    del squeeze_199
    del unsqueeze_702
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf210 = aten.convolution_backward(buf209, relu_65, primals_255, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf209
    del primals_255
    buf211 = buf210[0]
    buf212 = buf210[1]
    del buf210
    buf213 = buf207; del buf207  # reuse
    buf214 = empty((192, ), device='cpu', dtype=torch.float32)
    buf215 = empty((192, ), device='cpu', dtype=torch.float32)
    buf216 = buf211; del buf211  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29(c_void_p(buf216.data_ptr()), c_void_p(relu_65.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_714.data_ptr()), c_void_p(squeeze_196.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    del convolution_65
    del primals_131
    del relu_65
    del squeeze_196
    del unsqueeze_714
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf217 = aten.convolution_backward(buf216, relu_64, primals_254, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf216
    del primals_254
    buf218 = buf217[0]
    buf219 = buf217[1]
    del buf217
    buf220 = buf214; del buf214  # reuse
    buf221 = empty((192, ), device='cpu', dtype=torch.float32)
    buf222 = empty((192, ), device='cpu', dtype=torch.float32)
    buf223 = buf218; del buf218  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30(c_void_p(buf223.data_ptr()), c_void_p(relu_64.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(squeeze_193.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    del convolution_64
    del primals_129
    del relu_64
    del squeeze_193
    del unsqueeze_726
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf224 = aten.convolution_backward(buf223, cat_6, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_253
    buf225 = buf224[0]
    buf226 = buf224[1]
    del buf224
    buf227 = buf221; del buf221  # reuse
    buf228 = empty((192, ), device='cpu', dtype=torch.float32)
    buf229 = buf223; del buf223  # reuse
    buf231 = buf229; del buf229  # reuse
    buf230 = buf228; del buf228  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31(c_void_p(buf231.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(le_30.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(unsqueeze_738.data_ptr()), c_void_p(squeeze_190.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf227.data_ptr()))
    del convolution_63
    del le_30
    del primals_127
    del squeeze_190
    del unsqueeze_738
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf232 = aten.convolution_backward(buf231, relu_62, primals_252, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf231
    del primals_252
    buf233 = buf232[0]
    buf234 = buf232[1]
    del buf232
    buf235 = empty((192, ), device='cpu', dtype=torch.float32)
    buf236 = empty((192, ), device='cpu', dtype=torch.float32)
    buf237 = empty((192, ), device='cpu', dtype=torch.float32)
    buf238 = buf233; del buf233  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(buf238.data_ptr()), c_void_p(relu_62.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(unsqueeze_750.data_ptr()), c_void_p(squeeze_187.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    del convolution_62
    del primals_125
    del relu_62
    del squeeze_187
    del unsqueeze_750
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf239 = aten.convolution_backward(buf238, relu_61, primals_251, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf238
    del primals_251
    buf240 = buf239[0]
    buf241 = buf239[1]
    del buf239
    buf242 = buf236; del buf236  # reuse
    buf243 = empty((192, ), device='cpu', dtype=torch.float32)
    buf244 = empty((192, ), device='cpu', dtype=torch.float32)
    buf245 = buf240; del buf240  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33(c_void_p(buf245.data_ptr()), c_void_p(relu_61.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_762.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()))
    del convolution_61
    del primals_123
    del relu_61
    del squeeze_184
    del unsqueeze_762
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf246 = aten.convolution_backward(buf245, cat_6, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_250
    buf247 = buf246[0]
    buf248 = buf246[1]
    del buf246
    buf249 = buf243; del buf243  # reuse
    buf250 = empty((192, ), device='cpu', dtype=torch.float32)
    buf251 = buf245; del buf245  # reuse
    buf253 = buf251; del buf251  # reuse
    buf252 = buf250; del buf250  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34(c_void_p(buf253.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(le_33.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_774.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf249.data_ptr()))
    del buf139
    del buf166
    del convolution_60
    del le_33
    del primals_121
    del squeeze_181
    del unsqueeze_774
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf254 = aten.convolution_backward(buf253, cat_6, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_6
    del primals_249
    buf255 = buf254[0]
    buf256 = buf254[1]
    del buf254
    buf257 = buf253; del buf253  # reuse
    buf258 = empty((192, ), device='cpu', dtype=torch.float32)
    buf259 = empty((192, ), device='cpu', dtype=torch.float32)
    buf260 = empty((192, ), device='cpu', dtype=torch.float32)
    buf261 = buf257; del buf257  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35(c_void_p(buf261.data_ptr()), c_void_p(le_34.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_786.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del convolution_59
    del le_34
    del primals_119
    del squeeze_178
    del unsqueeze_786
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf262 = aten.convolution_backward(buf261, avg_pool2d_5, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_5
    del primals_248
    buf263 = buf262[0]
    buf264 = buf262[1]
    del buf262
    buf265 = reinterpret_tensor(buf180, (8, 768, 17, 17), (221952, 289, 17, 1), 0); del buf180  # reuse
    buf266 = buf261; del buf261  # reuse
    buf267 = buf259; del buf259  # reuse
    buf268 = empty((192, ), device='cpu', dtype=torch.float32)
    buf269 = empty((192, ), device='cpu', dtype=torch.float32)
    buf270 = buf266; del buf266  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf270.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(le_35.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(unsqueeze_798.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()))
    del buf263
    del convolution_58
    del le_35
    del primals_117
    del squeeze_175
    del unsqueeze_798
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf271 = aten.convolution_backward(buf270, relu_57, primals_247, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_247
    buf272 = buf271[0]
    buf273 = buf271[1]
    del buf271
    buf274 = empty((160, ), device='cpu', dtype=torch.float32)
    buf275 = empty((160, ), device='cpu', dtype=torch.float32)
    buf276 = empty((160, ), device='cpu', dtype=torch.float32)
    buf277 = buf272; del buf272  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37(c_void_p(buf277.data_ptr()), c_void_p(relu_57.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(unsqueeze_810.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    del convolution_57
    del primals_115
    del relu_57
    del squeeze_172
    del unsqueeze_810
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf278 = aten.convolution_backward(buf277, relu_56, primals_246, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf277
    del primals_246
    buf279 = buf278[0]
    buf280 = buf278[1]
    del buf278
    buf281 = buf275; del buf275  # reuse
    buf282 = empty((160, ), device='cpu', dtype=torch.float32)
    buf283 = empty((160, ), device='cpu', dtype=torch.float32)
    buf284 = buf279; del buf279  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_38(c_void_p(buf284.data_ptr()), c_void_p(relu_56.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_822.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    del convolution_56
    del primals_113
    del relu_56
    del squeeze_169
    del unsqueeze_822
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf285 = aten.convolution_backward(buf284, relu_55, primals_245, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf284
    del primals_245
    buf286 = buf285[0]
    buf287 = buf285[1]
    del buf285
    buf288 = buf282; del buf282  # reuse
    buf289 = empty((160, ), device='cpu', dtype=torch.float32)
    buf290 = empty((160, ), device='cpu', dtype=torch.float32)
    buf291 = buf286; del buf286  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(buf291.data_ptr()), c_void_p(relu_55.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_834.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()))
    del convolution_55
    del primals_111
    del relu_55
    del squeeze_166
    del unsqueeze_834
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf292 = aten.convolution_backward(buf291, relu_54, primals_244, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf291
    del primals_244
    buf293 = buf292[0]
    buf294 = buf292[1]
    del buf292
    buf295 = buf289; del buf289  # reuse
    buf296 = empty((160, ), device='cpu', dtype=torch.float32)
    buf297 = empty((160, ), device='cpu', dtype=torch.float32)
    buf298 = buf293; del buf293  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40(c_void_p(buf298.data_ptr()), c_void_p(relu_54.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_846.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    del convolution_54
    del primals_109
    del relu_54
    del squeeze_163
    del unsqueeze_846
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf299 = aten.convolution_backward(buf298, cat_5, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf298
    del primals_243
    buf300 = buf299[0]
    buf301 = buf299[1]
    del buf299
    buf302 = buf270; del buf270  # reuse
    buf303 = buf268; del buf268  # reuse
    buf304 = empty((192, ), device='cpu', dtype=torch.float32)
    buf305 = empty((192, ), device='cpu', dtype=torch.float32)
    buf306 = buf302; del buf302  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41(c_void_p(buf306.data_ptr()), c_void_p(le_40.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_858.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    del convolution_53
    del le_40
    del primals_107
    del squeeze_160
    del unsqueeze_858
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf307 = aten.convolution_backward(buf306, relu_52, primals_242, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_242
    buf308 = buf307[0]
    buf309 = buf307[1]
    del buf307
    buf310 = buf296; del buf296  # reuse
    buf311 = empty((160, ), device='cpu', dtype=torch.float32)
    buf312 = empty((160, ), device='cpu', dtype=torch.float32)
    buf313 = buf308; del buf308  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42(c_void_p(buf313.data_ptr()), c_void_p(relu_52.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_870.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del convolution_52
    del primals_105
    del relu_52
    del squeeze_157
    del unsqueeze_870
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf314 = aten.convolution_backward(buf313, relu_51, primals_241, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf313
    del primals_241
    buf315 = buf314[0]
    buf316 = buf314[1]
    del buf314
    buf317 = buf311; del buf311  # reuse
    buf318 = empty((160, ), device='cpu', dtype=torch.float32)
    buf319 = empty((160, ), device='cpu', dtype=torch.float32)
    buf320 = buf315; del buf315  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(buf320.data_ptr()), c_void_p(relu_51.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_882.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    del convolution_51
    del primals_103
    del relu_51
    del squeeze_154
    del unsqueeze_882
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf321 = aten.convolution_backward(buf320, cat_5, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf320
    del primals_240
    buf322 = buf321[0]
    buf323 = buf321[1]
    del buf321
    buf324 = buf306; del buf306  # reuse
    buf325 = buf304; del buf304  # reuse
    buf326 = empty((192, ), device='cpu', dtype=torch.float32)
    buf327 = empty((192, ), device='cpu', dtype=torch.float32)
    buf328 = buf324; del buf324  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_44(c_void_p(buf328.data_ptr()), c_void_p(le_43.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_894.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    del buf190
    del buf225
    del buf247
    del convolution_50
    del le_43
    del primals_101
    del squeeze_151
    del unsqueeze_894
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf329 = aten.convolution_backward(buf328, cat_5, primals_239, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_5
    del primals_239
    buf330 = buf329[0]
    buf331 = buf329[1]
    del buf329
    buf332 = buf328; del buf328  # reuse
    buf333 = buf326; del buf326  # reuse
    buf334 = empty((192, ), device='cpu', dtype=torch.float32)
    buf335 = empty((192, ), device='cpu', dtype=torch.float32)
    buf336 = buf332; del buf332  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_45(c_void_p(buf336.data_ptr()), c_void_p(le_44.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_906.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()))
    del convolution_49
    del le_44
    del primals_99
    del squeeze_148
    del unsqueeze_906
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf337 = aten.convolution_backward(buf336, avg_pool2d_4, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_4
    del primals_238
    buf338 = buf337[0]
    buf339 = buf337[1]
    del buf337
    buf340 = reinterpret_tensor(buf255, (8, 768, 17, 17), (221952, 289, 17, 1), 0); del buf255  # reuse
    buf341 = buf336; del buf336  # reuse
    buf342 = buf334; del buf334  # reuse
    buf343 = empty((192, ), device='cpu', dtype=torch.float32)
    buf344 = empty((192, ), device='cpu', dtype=torch.float32)
    buf345 = buf341; del buf341  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_46(c_void_p(buf345.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(le_45.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_918.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    del buf338
    del convolution_48
    del le_45
    del primals_97
    del squeeze_145
    del unsqueeze_918
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf346 = aten.convolution_backward(buf345, relu_47, primals_237, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_237
    buf347 = buf346[0]
    buf348 = buf346[1]
    del buf346
    buf349 = buf318; del buf318  # reuse
    buf350 = empty((160, ), device='cpu', dtype=torch.float32)
    buf351 = empty((160, ), device='cpu', dtype=torch.float32)
    buf352 = buf347; del buf347  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47(c_void_p(buf352.data_ptr()), c_void_p(relu_47.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_930.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    del convolution_47
    del primals_95
    del relu_47
    del squeeze_142
    del unsqueeze_930
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf353 = aten.convolution_backward(buf352, relu_46, primals_236, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf352
    del primals_236
    buf354 = buf353[0]
    buf355 = buf353[1]
    del buf353
    buf356 = buf350; del buf350  # reuse
    buf357 = empty((160, ), device='cpu', dtype=torch.float32)
    buf358 = empty((160, ), device='cpu', dtype=torch.float32)
    buf359 = buf354; del buf354  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_48(c_void_p(buf359.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_942.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    del convolution_46
    del primals_93
    del relu_46
    del squeeze_139
    del unsqueeze_942
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf360 = aten.convolution_backward(buf359, relu_45, primals_235, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf359
    del primals_235
    buf361 = buf360[0]
    buf362 = buf360[1]
    del buf360
    buf363 = buf357; del buf357  # reuse
    buf364 = empty((160, ), device='cpu', dtype=torch.float32)
    buf365 = empty((160, ), device='cpu', dtype=torch.float32)
    buf366 = buf361; del buf361  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49(c_void_p(buf366.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_954.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    del convolution_45
    del primals_91
    del relu_45
    del squeeze_136
    del unsqueeze_954
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf367 = aten.convolution_backward(buf366, relu_44, primals_234, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf366
    del primals_234
    buf368 = buf367[0]
    buf369 = buf367[1]
    del buf367
    buf370 = buf364; del buf364  # reuse
    buf371 = empty((160, ), device='cpu', dtype=torch.float32)
    buf372 = empty((160, ), device='cpu', dtype=torch.float32)
    buf373 = buf368; del buf368  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50(c_void_p(buf373.data_ptr()), c_void_p(relu_44.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_966.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    del convolution_44
    del primals_89
    del relu_44
    del squeeze_133
    del unsqueeze_966
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf374 = aten.convolution_backward(buf373, cat_4, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf373
    del primals_233
    buf375 = buf374[0]
    buf376 = buf374[1]
    del buf374
    buf377 = buf345; del buf345  # reuse
    buf378 = buf343; del buf343  # reuse
    buf379 = empty((192, ), device='cpu', dtype=torch.float32)
    buf380 = empty((192, ), device='cpu', dtype=torch.float32)
    buf381 = buf377; del buf377  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51(c_void_p(buf381.data_ptr()), c_void_p(le_50.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_978.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    del convolution_43
    del le_50
    del primals_87
    del squeeze_130
    del unsqueeze_978
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf382 = aten.convolution_backward(buf381, relu_42, primals_232, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_232
    buf383 = buf382[0]
    buf384 = buf382[1]
    del buf382
    buf385 = buf371; del buf371  # reuse
    buf386 = empty((160, ), device='cpu', dtype=torch.float32)
    buf387 = empty((160, ), device='cpu', dtype=torch.float32)
    buf388 = buf383; del buf383  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52(c_void_p(buf388.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_990.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()))
    del convolution_42
    del primals_85
    del relu_42
    del squeeze_127
    del unsqueeze_990
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf389 = aten.convolution_backward(buf388, relu_41, primals_231, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf388
    del primals_231
    buf390 = buf389[0]
    buf391 = buf389[1]
    del buf389
    buf392 = buf386; del buf386  # reuse
    buf393 = empty((160, ), device='cpu', dtype=torch.float32)
    buf394 = empty((160, ), device='cpu', dtype=torch.float32)
    buf395 = buf390; del buf390  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_53(c_void_p(buf395.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_1002.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    del buf393
    del convolution_41
    del primals_83
    del relu_41
    del squeeze_124
    del unsqueeze_1002
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf396 = aten.convolution_backward(buf395, cat_4, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf395
    del primals_230
    buf397 = buf396[0]
    buf398 = buf396[1]
    del buf396
    buf399 = buf381; del buf381  # reuse
    buf400 = buf379; del buf379  # reuse
    buf401 = empty((192, ), device='cpu', dtype=torch.float32)
    buf402 = empty((192, ), device='cpu', dtype=torch.float32)
    buf403 = buf399; del buf399  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54(c_void_p(buf403.data_ptr()), c_void_p(le_53.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_1014.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()))
    del buf265
    del buf300
    del buf322
    del convolution_40
    del le_53
    del primals_81
    del squeeze_121
    del unsqueeze_1014
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf404 = aten.convolution_backward(buf403, cat_4, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_4
    del primals_229
    buf405 = buf404[0]
    buf406 = buf404[1]
    del buf404
    buf407 = buf403; del buf403  # reuse
    buf408 = buf401; del buf401  # reuse
    buf409 = empty((192, ), device='cpu', dtype=torch.float32)
    buf410 = empty((192, ), device='cpu', dtype=torch.float32)
    buf411 = buf407; del buf407  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55(c_void_p(buf411.data_ptr()), c_void_p(le_54.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_1026.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()))
    del convolution_39
    del le_54
    del primals_79
    del squeeze_118
    del unsqueeze_1026
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf412 = aten.convolution_backward(buf411, avg_pool2d_3, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_3
    del primals_228
    buf413 = buf412[0]
    buf414 = buf412[1]
    del buf412
    buf415 = reinterpret_tensor(buf330, (8, 768, 17, 17), (221952, 289, 17, 1), 0); del buf330  # reuse
    buf416 = buf411; del buf411  # reuse
    buf417 = buf409; del buf409  # reuse
    buf418 = empty((192, ), device='cpu', dtype=torch.float32)
    buf419 = empty((192, ), device='cpu', dtype=torch.float32)
    buf420 = buf416; del buf416  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_56(c_void_p(buf420.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(le_55.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_1038.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()))
    del buf413
    del convolution_38
    del le_55
    del primals_77
    del squeeze_115
    del unsqueeze_1038
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf421 = aten.convolution_backward(buf420, relu_37, primals_227, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_227
    buf422 = buf421[0]
    buf423 = buf421[1]
    del buf421
    buf424 = empty((128, ), device='cpu', dtype=torch.float32)
    buf425 = empty((128, ), device='cpu', dtype=torch.float32)
    buf426 = empty((128, ), device='cpu', dtype=torch.float32)
    buf427 = buf422; del buf422  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_57(c_void_p(buf427.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_1050.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()))
    del convolution_37
    del primals_75
    del relu_37
    del squeeze_112
    del unsqueeze_1050
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf428 = aten.convolution_backward(buf427, relu_36, primals_226, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf427
    del primals_226
    buf429 = buf428[0]
    buf430 = buf428[1]
    del buf428
    buf431 = buf425; del buf425  # reuse
    buf432 = empty((128, ), device='cpu', dtype=torch.float32)
    buf433 = empty((128, ), device='cpu', dtype=torch.float32)
    buf434 = buf429; del buf429  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_58(c_void_p(buf434.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_1062.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()))
    del convolution_36
    del primals_73
    del relu_36
    del squeeze_109
    del unsqueeze_1062
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf435 = aten.convolution_backward(buf434, relu_35, primals_225, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf434
    del primals_225
    buf436 = buf435[0]
    buf437 = buf435[1]
    del buf435
    buf438 = buf432; del buf432  # reuse
    buf439 = empty((128, ), device='cpu', dtype=torch.float32)
    buf440 = empty((128, ), device='cpu', dtype=torch.float32)
    buf441 = buf436; del buf436  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_59(c_void_p(buf441.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_1074.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()))
    del convolution_35
    del primals_71
    del relu_35
    del squeeze_106
    del unsqueeze_1074
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf442 = aten.convolution_backward(buf441, relu_34, primals_224, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf441
    del primals_224
    buf443 = buf442[0]
    buf444 = buf442[1]
    del buf442
    buf445 = buf439; del buf439  # reuse
    buf446 = empty((128, ), device='cpu', dtype=torch.float32)
    buf447 = empty((128, ), device='cpu', dtype=torch.float32)
    buf448 = buf443; del buf443  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60(c_void_p(buf448.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_1086.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()))
    del convolution_34
    del primals_69
    del relu_34
    del squeeze_103
    del unsqueeze_1086
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf449 = aten.convolution_backward(buf448, cat_3, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf448
    del primals_223
    buf450 = buf449[0]
    buf451 = buf449[1]
    del buf449
    buf452 = buf420; del buf420  # reuse
    buf453 = buf418; del buf418  # reuse
    buf454 = empty((192, ), device='cpu', dtype=torch.float32)
    buf455 = empty((192, ), device='cpu', dtype=torch.float32)
    buf456 = buf452; del buf452  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61(c_void_p(buf456.data_ptr()), c_void_p(le_60.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_1098.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()))
    del convolution_33
    del le_60
    del primals_67
    del squeeze_100
    del unsqueeze_1098
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf457 = aten.convolution_backward(buf456, relu_32, primals_222, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_222
    buf458 = buf457[0]
    buf459 = buf457[1]
    del buf457
    buf460 = buf446; del buf446  # reuse
    buf461 = empty((128, ), device='cpu', dtype=torch.float32)
    buf462 = empty((128, ), device='cpu', dtype=torch.float32)
    buf463 = buf458; del buf458  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62(c_void_p(buf463.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_1110.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()))
    del convolution_32
    del primals_65
    del relu_32
    del squeeze_97
    del unsqueeze_1110
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf464 = aten.convolution_backward(buf463, relu_31, primals_221, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf463
    del primals_221
    buf465 = buf464[0]
    buf466 = buf464[1]
    del buf464
    buf467 = buf461; del buf461  # reuse
    buf468 = empty((128, ), device='cpu', dtype=torch.float32)
    buf469 = empty((128, ), device='cpu', dtype=torch.float32)
    buf470 = buf465; del buf465  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_63(c_void_p(buf470.data_ptr()), c_void_p(relu_31.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_1122.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()))
    del buf468
    del convolution_31
    del primals_63
    del relu_31
    del squeeze_94
    del unsqueeze_1122
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf471 = aten.convolution_backward(buf470, cat_3, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf470
    del primals_220
    buf472 = buf471[0]
    buf473 = buf471[1]
    del buf471
    buf474 = buf456; del buf456  # reuse
    buf475 = buf454; del buf454  # reuse
    buf476 = empty((192, ), device='cpu', dtype=torch.float32)
    buf477 = empty((192, ), device='cpu', dtype=torch.float32)
    buf478 = buf474; del buf474  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_64(c_void_p(buf478.data_ptr()), c_void_p(le_63.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_1134.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    del buf340
    del buf375
    del buf397
    del buf405
    del convolution_30
    del le_63
    del primals_61
    del squeeze_91
    del unsqueeze_1134
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf479 = aten.convolution_backward(buf478, cat_3, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf478
    del cat_3
    del primals_219
    buf480 = buf479[0]
    buf481 = buf479[1]
    del buf479
    buf482 = buf415; del buf415  # reuse
    cpp_fused_add_65(c_void_p(buf482.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf480.data_ptr()))
    del buf450
    del buf472
    del buf480
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf483 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf482, (8, 288, 17, 17), (221952, 289, 17, 1), 138720), cat_2, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_65)
    del getitem_65
    buf484 = buf483
    del buf483
    buf485 = empty((96, ), device='cpu', dtype=torch.float32)
    buf486 = empty((96, ), device='cpu', dtype=torch.float32)
    buf487 = empty((96, ), device='cpu', dtype=torch.float32)
    buf488 = empty_strided((8, 96, 17, 17), (27744, 1, 1632, 96), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_66(c_void_p(le_64.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_1146.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()))
    del convolution_29
    del le_64
    del primals_59
    del squeeze_88
    del unsqueeze_1146
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf489 = aten.convolution_backward(buf488, relu_28, primals_218, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf488
    del primals_218
    buf490 = buf489[0]
    buf491 = buf489[1]
    del buf489
    buf492 = buf486; del buf486  # reuse
    buf493 = empty((96, ), device='cpu', dtype=torch.float32)
    buf494 = empty((96, ), device='cpu', dtype=torch.float32)
    buf495 = buf490; del buf490  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67(c_void_p(buf495.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_1158.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()))
    del convolution_28
    del primals_57
    del relu_28
    del squeeze_85
    del unsqueeze_1158
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf496 = aten.convolution_backward(buf495, relu_27, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_217
    buf497 = buf496[0]
    buf498 = buf496[1]
    del buf496
    buf499 = empty((64, ), device='cpu', dtype=torch.float32)
    buf500 = empty((64, ), device='cpu', dtype=torch.float32)
    buf501 = empty((64, ), device='cpu', dtype=torch.float32)
    buf502 = buf497; del buf497  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68(c_void_p(buf502.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_1170.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()))
    del convolution_27
    del primals_55
    del relu_27
    del squeeze_82
    del unsqueeze_1170
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf503 = aten.convolution_backward(buf502, cat_2, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_216
    buf504 = buf503[0]
    buf505 = buf503[1]
    del buf503
    buf506 = empty((384, ), device='cpu', dtype=torch.float32)
    buf507 = empty((384, ), device='cpu', dtype=torch.float32)
    buf508 = empty((384, ), device='cpu', dtype=torch.float32)
    buf509 = empty_strided((8, 384, 17, 17), (110976, 1, 6528, 384), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_69(c_void_p(le_67.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_1182.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()))
    del buf482
    del buf507
    del convolution_26
    del le_67
    del primals_53
    del squeeze_79
    del unsqueeze_1182
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf510 = aten.convolution_backward(buf509, cat_2, primals_215, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf509
    del cat_2
    del primals_215
    buf511 = buf510[0]
    buf512 = buf510[1]
    del buf510
    buf513 = buf500; del buf500  # reuse
    buf514 = empty((64, ), device='cpu', dtype=torch.float32)
    buf515 = buf502; del buf502  # reuse
    buf517 = buf515; del buf515  # reuse
    buf516 = buf514; del buf514  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70(c_void_p(buf517.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(le_68.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_1194.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf513.data_ptr()))
    del convolution_25
    del le_68
    del primals_51
    del squeeze_76
    del unsqueeze_1194
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf518 = aten.convolution_backward(buf517, avg_pool2d_2, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_2
    del buf517
    del primals_214
    buf519 = buf518[0]
    buf520 = buf518[1]
    del buf518
    buf521 = empty((8, 288, 35, 35), device='cpu', dtype=torch.float32)
    buf522 = buf493; del buf493  # reuse
    buf523 = empty((96, ), device='cpu', dtype=torch.float32)
    buf524 = buf495; del buf495  # reuse
    buf526 = buf524; del buf524  # reuse
    buf525 = buf523; del buf523  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_71(c_void_p(buf526.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(le_69.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_1206.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()))
    del buf519
    del convolution_24
    del le_69
    del primals_49
    del squeeze_73
    del unsqueeze_1206
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf527 = aten.convolution_backward(buf526, relu_23, primals_213, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf526
    del primals_213
    buf528 = buf527[0]
    buf529 = buf527[1]
    del buf527
    buf530 = empty((96, ), device='cpu', dtype=torch.float32)
    buf531 = empty((96, ), device='cpu', dtype=torch.float32)
    buf532 = empty((96, ), device='cpu', dtype=torch.float32)
    buf533 = buf528; del buf528  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_72(c_void_p(buf533.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_1218.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    del convolution_23
    del primals_47
    del relu_23
    del squeeze_70
    del unsqueeze_1218
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf534 = aten.convolution_backward(buf533, relu_22, primals_212, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_212
    buf535 = buf534[0]
    buf536 = buf534[1]
    del buf534
    buf537 = empty((64, ), device='cpu', dtype=torch.float32)
    buf538 = empty((64, ), device='cpu', dtype=torch.float32)
    buf539 = empty((64, ), device='cpu', dtype=torch.float32)
    buf540 = buf535; del buf535  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_73(c_void_p(buf540.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_1230.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()))
    del convolution_22
    del primals_45
    del relu_22
    del squeeze_67
    del unsqueeze_1230
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf541 = aten.convolution_backward(buf540, cat_1, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_211
    buf542 = buf541[0]
    buf543 = buf541[1]
    del buf541
    buf544 = buf538; del buf538  # reuse
    buf545 = empty((64, ), device='cpu', dtype=torch.float32)
    buf546 = buf540; del buf540  # reuse
    buf548 = buf546; del buf546  # reuse
    buf547 = buf545; del buf545  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_74(c_void_p(buf548.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(le_72.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_1242.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf544.data_ptr()))
    del convolution_21
    del le_72
    del primals_43
    del squeeze_64
    del unsqueeze_1242
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf549 = aten.convolution_backward(buf548, relu_20, primals_210, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_210
    buf550 = buf549[0]
    buf551 = buf549[1]
    del buf549
    buf552 = empty((48, ), device='cpu', dtype=torch.float32)
    buf553 = empty((48, ), device='cpu', dtype=torch.float32)
    buf554 = empty((48, ), device='cpu', dtype=torch.float32)
    buf555 = buf550; del buf550  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_75(c_void_p(buf555.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_1254.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()))
    del convolution_20
    del primals_41
    del relu_20
    del squeeze_61
    del unsqueeze_1254
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf556 = aten.convolution_backward(buf555, cat_1, primals_209, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf555
    del primals_209
    buf557 = buf556[0]
    buf558 = buf556[1]
    del buf556
    buf559 = empty((64, ), device='cpu', dtype=torch.float32)
    buf560 = empty((64, ), device='cpu', dtype=torch.float32)
    buf561 = buf548; del buf548  # reuse
    buf563 = buf561; del buf561  # reuse
    buf562 = buf560; del buf560  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76(c_void_p(buf563.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(le_74.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_1266.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf559.data_ptr()))
    del buf484
    del buf504
    del buf511
    del convolution_19
    del le_74
    del primals_39
    del squeeze_58
    del unsqueeze_1266
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf564 = aten.convolution_backward(buf563, cat_1, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del cat_1
    del primals_208
    buf565 = buf564[0]
    buf566 = buf564[1]
    del buf564
    buf567 = buf563; del buf563  # reuse
    buf568 = empty((64, ), device='cpu', dtype=torch.float32)
    buf569 = empty((64, ), device='cpu', dtype=torch.float32)
    buf570 = empty((64, ), device='cpu', dtype=torch.float32)
    buf571 = buf567; del buf567  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77(c_void_p(buf571.data_ptr()), c_void_p(le_75.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_1278.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()))
    del convolution_18
    del le_75
    del primals_37
    del squeeze_55
    del unsqueeze_1278
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf572 = aten.convolution_backward(buf571, avg_pool2d_1, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d_1
    del buf571
    del primals_207
    buf573 = buf572[0]
    buf574 = buf572[1]
    del buf572
    buf575 = empty((8, 256, 35, 35), device='cpu', dtype=torch.float32)
    buf576 = buf533; del buf533  # reuse
    buf577 = buf531; del buf531  # reuse
    buf578 = empty((96, ), device='cpu', dtype=torch.float32)
    buf579 = empty((96, ), device='cpu', dtype=torch.float32)
    buf580 = buf576; del buf576  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_78(c_void_p(buf580.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(le_76.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_1290.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()))
    del buf573
    del convolution_17
    del le_76
    del primals_35
    del squeeze_52
    del unsqueeze_1290
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf581 = aten.convolution_backward(buf580, relu_16, primals_206, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf580
    del primals_206
    buf582 = buf581[0]
    buf583 = buf581[1]
    del buf581
    buf584 = buf578; del buf578  # reuse
    buf585 = empty((96, ), device='cpu', dtype=torch.float32)
    buf586 = empty((96, ), device='cpu', dtype=torch.float32)
    buf587 = buf582; del buf582  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_79(c_void_p(buf587.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_1302.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()))
    del convolution_16
    del primals_33
    del relu_16
    del squeeze_49
    del unsqueeze_1302
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf588 = aten.convolution_backward(buf587, relu_15, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_205
    buf589 = buf588[0]
    buf590 = buf588[1]
    del buf588
    buf591 = buf569; del buf569  # reuse
    buf592 = empty((64, ), device='cpu', dtype=torch.float32)
    buf593 = empty((64, ), device='cpu', dtype=torch.float32)
    buf594 = buf589; del buf589  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_80(c_void_p(buf594.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_1314.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()))
    del convolution_15
    del primals_31
    del relu_15
    del squeeze_46
    del unsqueeze_1314
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf595 = aten.convolution_backward(buf594, cat, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_204
    buf596 = buf595[0]
    buf597 = buf595[1]
    del buf595
    buf598 = buf594; del buf594  # reuse
    buf599 = buf592; del buf592  # reuse
    buf600 = empty((64, ), device='cpu', dtype=torch.float32)
    buf601 = empty((64, ), device='cpu', dtype=torch.float32)
    buf602 = buf598; del buf598  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81(c_void_p(buf602.data_ptr()), c_void_p(le_79.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_1326.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()))
    del convolution_14
    del le_79
    del primals_29
    del squeeze_43
    del unsqueeze_1326
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf603 = aten.convolution_backward(buf602, relu_13, primals_203, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_203
    buf604 = buf603[0]
    buf605 = buf603[1]
    del buf603
    buf606 = buf553; del buf553  # reuse
    buf607 = empty((48, ), device='cpu', dtype=torch.float32)
    buf608 = empty((48, ), device='cpu', dtype=torch.float32)
    buf609 = buf604; del buf604  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_82(c_void_p(buf609.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_1338.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf608.data_ptr()))
    del convolution_13
    del primals_27
    del relu_13
    del squeeze_40
    del unsqueeze_1338
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf610 = aten.convolution_backward(buf609, cat, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf609
    del primals_202
    buf611 = buf610[0]
    buf612 = buf610[1]
    del buf610
    buf613 = buf602; del buf602  # reuse
    buf614 = buf600; del buf600  # reuse
    buf615 = empty((64, ), device='cpu', dtype=torch.float32)
    buf616 = empty((64, ), device='cpu', dtype=torch.float32)
    buf617 = buf613; del buf613  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83(c_void_p(buf617.data_ptr()), c_void_p(le_81.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_1350.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf616.data_ptr()))
    del buf521
    del buf542
    del buf557
    del buf565
    del convolution_12
    del le_81
    del primals_25
    del squeeze_37
    del unsqueeze_1350
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf618 = aten.convolution_backward(buf617, cat, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf617
    del cat
    del primals_201
    buf619 = buf618[0]
    buf620 = buf618[1]
    del buf618
    buf621 = empty_strided((8, 32, 35, 35), (39200, 1, 1120, 32), device='cpu', dtype=torch.float32)
    buf622 = empty((32, ), device='cpu', dtype=torch.float32)
    buf623 = empty((32, ), device='cpu', dtype=torch.float32)
    buf624 = empty((32, ), device='cpu', dtype=torch.float32)
    buf625 = buf621; del buf621  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_84(c_void_p(buf625.data_ptr()), c_void_p(le_82.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_1362.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()))
    del convolution_11
    del le_82
    del primals_23
    del squeeze_34
    del unsqueeze_1362
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf626 = aten.convolution_backward(buf625, avg_pool2d, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del avg_pool2d
    del buf625
    del primals_200
    buf627 = buf626[0]
    buf628 = buf626[1]
    del buf626
    buf629 = empty((8, 192, 35, 35), device='cpu', dtype=torch.float32)
    buf630 = buf587; del buf587  # reuse
    buf631 = buf585; del buf585  # reuse
    buf632 = empty((96, ), device='cpu', dtype=torch.float32)
    buf633 = empty((96, ), device='cpu', dtype=torch.float32)
    buf634 = buf630; del buf630  # reuse
    cpp_fused_avg_pool2d_backward_convolution_backward_native_batch_norm_backward_threshold_backward_85(c_void_p(buf634.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(le_83.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_1374.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf633.data_ptr()))
    del buf627
    del convolution_10
    del le_83
    del primals_21
    del squeeze_31
    del unsqueeze_1374
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf635 = aten.convolution_backward(buf634, relu_9, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf634
    del primals_199
    buf636 = buf635[0]
    buf637 = buf635[1]
    del buf635
    buf638 = buf632; del buf632  # reuse
    buf639 = empty((96, ), device='cpu', dtype=torch.float32)
    buf640 = empty((96, ), device='cpu', dtype=torch.float32)
    buf641 = buf636; del buf636  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_86(c_void_p(buf641.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_1386.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf640.data_ptr()))
    del buf639
    del convolution_9
    del primals_19
    del relu_9
    del squeeze_28
    del unsqueeze_1386
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf642 = aten.convolution_backward(buf641, relu_8, primals_198, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf641
    del primals_198
    buf643 = buf642[0]
    buf644 = buf642[1]
    del buf642
    buf645 = buf615; del buf615  # reuse
    buf646 = empty((64, ), device='cpu', dtype=torch.float32)
    buf647 = empty((64, ), device='cpu', dtype=torch.float32)
    buf648 = buf643; del buf643  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_87(c_void_p(buf648.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_1398.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()))
    del convolution_8
    del primals_17
    del relu_8
    del squeeze_25
    del unsqueeze_1398
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf649 = aten.convolution_backward(buf648, getitem_12, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_197
    buf650 = buf649[0]
    buf651 = buf649[1]
    del buf649
    buf652 = buf648; del buf648  # reuse
    buf653 = buf646; del buf646  # reuse
    buf654 = empty((64, ), device='cpu', dtype=torch.float32)
    buf655 = empty((64, ), device='cpu', dtype=torch.float32)
    buf656 = buf652; del buf652  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_88(c_void_p(buf656.data_ptr()), c_void_p(le_86.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_1410.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf655.data_ptr()))
    del convolution_7
    del le_86
    del primals_15
    del squeeze_22
    del unsqueeze_1410
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf657 = aten.convolution_backward(buf656, relu_6, primals_196, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_196
    buf658 = buf657[0]
    buf659 = buf657[1]
    del buf657
    buf660 = buf607; del buf607  # reuse
    buf661 = empty((48, ), device='cpu', dtype=torch.float32)
    buf662 = empty((48, ), device='cpu', dtype=torch.float32)
    buf663 = buf658; del buf658  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_89(c_void_p(buf663.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_1422.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf662.data_ptr()))
    del buf661
    del convolution_6
    del primals_13
    del relu_6
    del squeeze_19
    del unsqueeze_1422
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf664 = aten.convolution_backward(buf663, getitem_12, primals_195, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf663
    del primals_195
    buf665 = buf664[0]
    buf666 = buf664[1]
    del buf664
    buf667 = buf656; del buf656  # reuse
    buf668 = buf654; del buf654  # reuse
    buf669 = empty((64, ), device='cpu', dtype=torch.float32)
    buf670 = empty((64, ), device='cpu', dtype=torch.float32)
    buf671 = buf667; del buf667  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_90(c_void_p(buf671.data_ptr()), c_void_p(le_88.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_1434.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf670.data_ptr()))
    del buf575
    del buf596
    del buf611
    del buf619
    del convolution_5
    del le_88
    del primals_11
    del squeeze_16
    del unsqueeze_1434
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf672 = aten.convolution_backward(buf671, getitem_12, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf671
    del getitem_12
    del primals_194
    buf673 = buf672[0]
    buf674 = buf672[1]
    del buf672
    buf675 = buf650; del buf650  # reuse
    cpp_fused_add_91(c_void_p(buf675.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf673.data_ptr()))
    del buf629
    del buf665
    del buf673
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf676 = aten.max_pool2d_with_indices_backward(buf675, relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_13)
    del buf675
    del getitem_13
    buf677 = buf676
    del buf676
    buf678 = buf476; del buf476  # reuse
    buf679 = empty((192, ), device='cpu', dtype=torch.float32)
    buf680 = empty((192, ), device='cpu', dtype=torch.float32)
    buf681 = buf677; del buf677  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92(c_void_p(buf681.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_1446.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf680.data_ptr()))
    del buf679
    del convolution_4
    del primals_9
    del relu_4
    del squeeze_13
    del unsqueeze_1446
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf682 = aten.convolution_backward(buf681, relu_3, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf681
    del primals_193
    buf683 = buf682[0]
    buf684 = buf682[1]
    del buf682
    buf685 = empty((80, ), device='cpu', dtype=torch.float32)
    buf686 = empty((80, ), device='cpu', dtype=torch.float32)
    buf687 = empty((80, ), device='cpu', dtype=torch.float32)
    buf688 = buf683; del buf683  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_93(c_void_p(buf688.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_1458.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf687.data_ptr()))
    del buf686
    del convolution_3
    del primals_7
    del relu_3
    del squeeze_10
    del unsqueeze_1458
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf689 = aten.convolution_backward(buf688, getitem_6, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf688
    del getitem_6
    del primals_192
    buf690 = buf689[0]
    buf691 = buf689[1]
    del buf689
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf692 = aten.max_pool2d_with_indices_backward(buf690, relu_2, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_7)
    del buf690
    del getitem_7
    buf693 = buf692
    del buf692
    buf694 = buf669; del buf669  # reuse
    buf695 = empty((64, ), device='cpu', dtype=torch.float32)
    buf696 = empty((64, ), device='cpu', dtype=torch.float32)
    buf697 = buf693; del buf693  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94(c_void_p(buf697.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_1470.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf696.data_ptr()))
    del buf695
    del convolution_2
    del primals_5
    del relu_2
    del squeeze_7
    del unsqueeze_1470
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf698 = aten.convolution_backward(buf697, relu_1, primals_191, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf697
    del primals_191
    buf699 = buf698[0]
    buf700 = buf698[1]
    del buf698
    buf701 = buf623; del buf623  # reuse
    buf702 = empty((32, ), device='cpu', dtype=torch.float32)
    buf703 = empty((32, ), device='cpu', dtype=torch.float32)
    buf704 = buf699; del buf699  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_95(c_void_p(buf704.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_1482.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf703.data_ptr()))
    del convolution_1
    del primals_3
    del relu_1
    del squeeze_4
    del unsqueeze_1482
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf705 = aten.convolution_backward(buf704, relu, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf704
    del primals_190
    buf706 = buf705[0]
    buf707 = buf705[1]
    del buf705
    buf708 = buf702; del buf702  # reuse
    buf709 = empty((32, ), device='cpu', dtype=torch.float32)
    buf710 = empty((32, ), device='cpu', dtype=torch.float32)
    buf711 = buf706; del buf706  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_96(c_void_p(buf711.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_1494.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf710.data_ptr()))
    del buf709
    del convolution
    del primals_1
    del relu
    del squeeze_1
    del unsqueeze_1494
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf712 = aten.convolution_backward(buf711, primals_567, primals_189, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf711
    del primals_189
    del primals_567
    buf713 = buf712[1]
    return (buf710, buf708, buf703, buf701, buf696, buf694, buf687, buf685, buf680, buf678, buf670, buf668, buf662, buf660, buf655, buf653, buf647, buf645, buf640, buf638, buf633, buf631, buf624, buf622, buf616, buf614, buf608, buf606, buf601, buf599, buf593, buf591, buf586, buf584, buf579, buf577, buf570, buf568, buf562, buf559, buf554, buf552, buf547, buf544, buf539, buf537, buf532, buf530, buf525, buf522, buf516, buf513, buf508, buf506, buf501, buf499, buf494, buf492, buf487, buf485, buf477, buf475, buf469, buf467, buf462, buf460, buf455, buf453, buf447, buf445, buf440, buf438, buf433, buf431, buf426, buf424, buf419, buf417, buf410, buf408, buf402, buf400, buf394, buf392, buf387, buf385, buf380, buf378, buf372, buf370, buf365, buf363, buf358, buf356, buf351, buf349, buf344, buf342, buf335, buf333, buf327, buf325, buf319, buf317, buf312, buf310, buf305, buf303, buf297, buf295, buf290, buf288, buf283, buf281, buf276, buf274, buf269, buf267, buf260, buf258, buf252, buf249, buf244, buf242, buf237, buf235, buf230, buf227, buf222, buf220, buf215, buf213, buf208, buf206, buf201, buf199, buf194, buf191, buf185, buf182, buf177, buf175, buf170, buf168, buf163, buf161, buf156, buf154, buf149, buf147, buf142, buf140, buf132, buf130, buf125, buf122, buf117, buf115, buf109, buf107, buf101, buf99, buf95, buf92, buf87, buf85, buf79, buf77, buf70, buf68, buf62, buf60, buf56, buf53, buf48, buf46, buf41, buf39, buf34, buf32, buf28, buf25, buf20, buf18, buf13, buf11, buf5, buf3, buf713, buf707, buf700, buf691, buf684, buf674, buf666, buf659, buf651, buf644, buf637, buf628, buf620, buf612, buf605, buf597, buf590, buf583, buf574, buf566, buf558, buf551, buf543, buf536, buf529, buf520, buf512, buf505, buf498, buf491, buf481, buf473, buf466, buf459, buf451, buf444, buf437, buf430, buf423, buf414, buf406, buf398, buf391, buf384, buf376, buf369, buf362, buf355, buf348, buf339, buf331, buf323, buf316, buf309, buf301, buf294, buf287, buf280, buf273, buf264, buf256, buf248, buf241, buf234, buf226, buf219, buf212, buf205, buf198, buf189, buf181, buf174, buf167, buf160, buf153, buf146, buf136, buf128, buf121, buf113, buf105, buf98, buf91, buf83, buf74, buf66, buf59, buf52, buf45, buf38, buf31, buf24, buf17, buf9, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((80, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((192, 80, 3, 3), (720, 1, 240, 80), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((384, 288, 3, 3), (2592, 1, 864, 288), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((192, 128, 7, 1), (896, 1, 128, 128), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((128, 128, 7, 1), (896, 1, 128, 128), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((128, 128, 7, 1), (896, 1, 128, 128), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((192, 128, 1, 7), (896, 1, 896, 128), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((192, 160, 1, 7), (1120, 1, 1120, 160), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((160, 160, 7, 1), (1120, 1, 160, 160), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((192, 160, 1, 7), (1120, 1, 1120, 160), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((320, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((192, 192, 7, 1), (1344, 1, 192, 192), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((384, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((448, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((192, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((320, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((384, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((448, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((384, 384, 3, 1), (1152, 1, 384, 384), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((192, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_567 = rand_strided((8, 3, 299, 299), (268203, 1, 897, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 149, 149), (710432, 1, 4768, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 32, 149, 149), (710432, 1, 4768, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 147, 147), (691488, 1, 4704, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 32, 147, 147), (691488, 1, 4704, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 147, 147), (1382976, 1, 9408, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 147, 147), (1382976, 1, 9408, 64), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((8, 64, 73, 73), (341056, 1, 4672, 64), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((8, 64, 73, 73), (341056, 1, 4672, 64), device='cpu', dtype=torch.int64)
    convolution_3 = rand_strided((8, 80, 73, 73), (426320, 1, 5840, 80), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 80, 73, 73), (426320, 1, 5840, 80), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 192, 71, 71), (967872, 1, 13632, 192), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 192, 71, 71), (967872, 1, 13632, 192), device='cpu', dtype=torch.float32)
    getitem_12 = rand_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cpu', dtype=torch.int64)
    convolution_5 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 32, 35, 35), (39200, 1, 1120, 32), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 256, 35, 35), (313600, 1, 8960, 256), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((8, 256, 35, 35), (313600, 1, 8960, 256), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 384, 17, 17), (110976, 1, 6528, 384), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 96, 17, 17), (27744, 1, 1632, 96), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_65 = rand_strided((8, 288, 17, 17), (83232, 1, 4896, 288), device='cpu', dtype=torch.int64)
    cat_3 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_36 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    avg_pool2d_3 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_41 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_42 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_44 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_46 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_47 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    avg_pool2d_4 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    cat_5 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_51 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_52 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_54 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_55 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_56 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_57 = rand_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    avg_pool2d_5 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    cat_6 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_61 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_187 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_62 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_190 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_193 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_64 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_196 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_65 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_199 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_66 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_202 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_67 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_205 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    avg_pool2d_6 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_208 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    cat_7 = rand_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_211 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_70 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.float32)
    squeeze_214 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_217 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_72 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_220 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_73 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    squeeze_223 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_74 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cpu', dtype=torch.float32)
    squeeze_226 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_159 = rand_strided((8, 768, 8, 8), (49152, 1, 6144, 768), device='cpu', dtype=torch.int64)
    cat_8 = rand_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.float32)
    squeeze_229 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_232 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_77 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    convolution_78 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_235 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_238 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cpu', dtype=torch.float32)
    squeeze_241 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    relu_80 = rand_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cpu', dtype=torch.float32)
    convolution_81 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_244 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_81 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    convolution_82 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_247 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_83 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_250 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    avg_pool2d_7 = rand_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cpu', dtype=torch.float32)
    convolution_84 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cpu', dtype=torch.float32)
    squeeze_253 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    cat_11 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    convolution_85 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.float32)
    squeeze_256 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_86 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_259 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_86 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    convolution_87 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_262 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_88 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_265 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_89 = rand_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cpu', dtype=torch.float32)
    squeeze_268 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    relu_89 = rand_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cpu', dtype=torch.float32)
    convolution_90 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_271 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    relu_90 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    convolution_91 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_274 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_92 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.float32)
    squeeze_277 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    avg_pool2d_8 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    convolution_93 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cpu', dtype=torch.float32)
    squeeze_280 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cpu', dtype=torch.bool)
    unsqueeze_378 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_1 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.bool)
    unsqueeze_390 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_2 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.bool)
    unsqueeze_402 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_5 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.bool)
    unsqueeze_438 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_6 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.bool)
    unsqueeze_450 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_8 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.bool)
    unsqueeze_474 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_9 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cpu', dtype=torch.bool)
    unsqueeze_486 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_10 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.bool)
    unsqueeze_498 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_11 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.bool)
    unsqueeze_510 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_14 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.bool)
    unsqueeze_546 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_15 = rand_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cpu', dtype=torch.bool)
    unsqueeze_558 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_17 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.bool)
    unsqueeze_582 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_18 = rand_strided((8, 192, 8, 8), (12288, 1, 1536, 192), device='cpu', dtype=torch.bool)
    unsqueeze_594 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_22 = rand_strided((8, 320, 8, 8), (20480, 1, 2560, 320), device='cpu', dtype=torch.bool)
    unsqueeze_642 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_24 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_666 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_25 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_678 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_30 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_738 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_33 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_774 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_34 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_786 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_35 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_798 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_822 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_40 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_858 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_870 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_882 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_43 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_894 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_44 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_906 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_45 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_918 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_930 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_942 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_954 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_966 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_50 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_978 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_990 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1002 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_53 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_1014 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_54 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_1026 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_55 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_1038 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1050 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1062 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1074 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1086 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_60 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_1098 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1110 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1122 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_63 = rand_strided((8, 192, 17, 17), (55488, 1, 3264, 192), device='cpu', dtype=torch.bool)
    unsqueeze_1134 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_64 = rand_strided((8, 96, 17, 17), (27744, 1, 1632, 96), device='cpu', dtype=torch.bool)
    unsqueeze_1146 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1158 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1170 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_67 = rand_strided((8, 384, 17, 17), (110976, 1, 6528, 384), device='cpu', dtype=torch.bool)
    unsqueeze_1182 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_68 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1194 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_69 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.bool)
    unsqueeze_1206 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1218 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1230 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_72 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1242 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1254 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_74 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1266 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_75 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1278 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_76 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.bool)
    unsqueeze_1290 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1302 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1314 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_79 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1326 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1338 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_81 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1350 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_82 = rand_strided((8, 32, 35, 35), (39200, 1, 1120, 32), device='cpu', dtype=torch.bool)
    unsqueeze_1362 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_83 = rand_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cpu', dtype=torch.bool)
    unsqueeze_1374 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1386 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1398 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_86 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1410 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1422 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_88 = rand_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cpu', dtype=torch.bool)
    unsqueeze_1434 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1446 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1458 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1470 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1482 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1494 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_567, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, getitem_12, getitem_13, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, avg_pool2d, convolution_11, squeeze_34, cat, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, avg_pool2d_1, convolution_18, squeeze_55, cat_1, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, avg_pool2d_2, convolution_25, squeeze_76, cat_2, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, getitem_65, cat_3, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, avg_pool2d_3, convolution_39, squeeze_118, cat_4, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_41, convolution_42, squeeze_127, relu_42, convolution_43, squeeze_130, convolution_44, squeeze_133, relu_44, convolution_45, squeeze_136, relu_45, convolution_46, squeeze_139, relu_46, convolution_47, squeeze_142, relu_47, convolution_48, squeeze_145, avg_pool2d_4, convolution_49, squeeze_148, cat_5, convolution_50, squeeze_151, convolution_51, squeeze_154, relu_51, convolution_52, squeeze_157, relu_52, convolution_53, squeeze_160, convolution_54, squeeze_163, relu_54, convolution_55, squeeze_166, relu_55, convolution_56, squeeze_169, relu_56, convolution_57, squeeze_172, relu_57, convolution_58, squeeze_175, avg_pool2d_5, convolution_59, squeeze_178, cat_6, convolution_60, squeeze_181, convolution_61, squeeze_184, relu_61, convolution_62, squeeze_187, relu_62, convolution_63, squeeze_190, convolution_64, squeeze_193, relu_64, convolution_65, squeeze_196, relu_65, convolution_66, squeeze_199, relu_66, convolution_67, squeeze_202, relu_67, convolution_68, squeeze_205, avg_pool2d_6, convolution_69, squeeze_208, cat_7, convolution_70, squeeze_211, relu_70, convolution_71, squeeze_214, convolution_72, squeeze_217, relu_72, convolution_73, squeeze_220, relu_73, convolution_74, squeeze_223, relu_74, convolution_75, squeeze_226, getitem_159, cat_8, convolution_76, squeeze_229, convolution_77, squeeze_232, relu_77, convolution_78, squeeze_235, convolution_79, squeeze_238, convolution_80, squeeze_241, relu_80, convolution_81, squeeze_244, relu_81, convolution_82, squeeze_247, convolution_83, squeeze_250, avg_pool2d_7, convolution_84, squeeze_253, cat_11, convolution_85, squeeze_256, convolution_86, squeeze_259, relu_86, convolution_87, squeeze_262, convolution_88, squeeze_265, convolution_89, squeeze_268, relu_89, convolution_90, squeeze_271, relu_90, convolution_91, squeeze_274, convolution_92, squeeze_277, avg_pool2d_8, convolution_93, squeeze_280, clone, permute_1, le, unsqueeze_378, le_1, unsqueeze_390, le_2, unsqueeze_402, unsqueeze_414, unsqueeze_426, le_5, unsqueeze_438, le_6, unsqueeze_450, unsqueeze_462, le_8, unsqueeze_474, le_9, unsqueeze_486, le_10, unsqueeze_498, le_11, unsqueeze_510, unsqueeze_522, unsqueeze_534, le_14, unsqueeze_546, le_15, unsqueeze_558, unsqueeze_570, le_17, unsqueeze_582, le_18, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, le_22, unsqueeze_642, unsqueeze_654, le_24, unsqueeze_666, le_25, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, le_30, unsqueeze_738, unsqueeze_750, unsqueeze_762, le_33, unsqueeze_774, le_34, unsqueeze_786, le_35, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_40, unsqueeze_858, unsqueeze_870, unsqueeze_882, le_43, unsqueeze_894, le_44, unsqueeze_906, le_45, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, le_50, unsqueeze_978, unsqueeze_990, unsqueeze_1002, le_53, unsqueeze_1014, le_54, unsqueeze_1026, le_55, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, unsqueeze_1074, unsqueeze_1086, le_60, unsqueeze_1098, unsqueeze_1110, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, unsqueeze_1230, le_72, unsqueeze_1242, unsqueeze_1254, le_74, unsqueeze_1266, le_75, unsqueeze_1278, le_76, unsqueeze_1290, unsqueeze_1302, unsqueeze_1314, le_79, unsqueeze_1326, unsqueeze_1338, le_81, unsqueeze_1350, le_82, unsqueeze_1362, le_83, unsqueeze_1374, unsqueeze_1386, unsqueeze_1398, le_86, unsqueeze_1410, unsqueeze_1422, le_88, unsqueeze_1434, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1482, unsqueeze_1494, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gluon_inception_v3', benchmark_compiled_module)
