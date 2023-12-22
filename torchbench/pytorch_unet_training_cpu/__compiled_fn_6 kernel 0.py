
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_0 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1227520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1227520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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


cpp_fused__unsafe_index_put_add_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_new_zeros_rsub_threshold_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const long* in_ptr6,
                       const bool* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr7 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19619840L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19619840L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19619840L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19619840L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr3 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(958L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1)];
                            auto tmp4 = in_ptr1[static_cast<long>(x2)];
                            auto tmp14 = in_ptr3[static_cast<long>(x2)];
                            auto tmp16 = in_ptr4[static_cast<long>(x1)];
                            auto tmp18 = in_ptr5[static_cast<long>(x1)];
                            auto tmp29 = in_ptr6[static_cast<long>(x2)];
                            auto tmp1 = decltype(tmp0)(tmp0 + 320);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            auto tmp5 = decltype(tmp4)(tmp4 + 479);
                            auto tmp6 = tmp4 < 0;
                            auto tmp7 = tmp6 ? tmp5 : tmp4;
                            auto tmp8 = c10::convert<long>(x2);
                            auto tmp9 = static_cast<long>(959);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr2[static_cast<long>(64L + x3 + (128L*x2) + (122752L*x1) + (78561280L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp19 = decltype(tmp18)(tmp18 + 320);
                            auto tmp20 = tmp18 < 0;
                            auto tmp21 = tmp20 ? tmp19 : tmp18;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr2[static_cast<long>(64L + x3 + (128L*x2) + (122752L*x1) + (78561280L*x0))];
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp10 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                            auto tmp25 = decltype(tmp24)(tmp24 * tmp14);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp16);
                            auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                            auto tmp30 = decltype(tmp29)(tmp29 + 479);
                            auto tmp31 = tmp29 < 0;
                            auto tmp32 = tmp31 ? tmp30 : tmp29;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr2[static_cast<long>(64L + x3 + (128L*x2) + (122752L*x1) + (78561280L*x0))];
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp10 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp36 = decltype(tmp26)(tmp26 - tmp14);
                            auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                            auto tmp38 = decltype(tmp37)(tmp37 * tmp16);
                            auto tmp39 = [&]
                            {
                                auto tmp40 = in_ptr2[static_cast<long>(64L + x3 + (128L*x2) + (122752L*x1) + (78561280L*x0))];
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp42 = decltype(tmp41)(tmp41 * tmp36);
                            auto tmp43 = decltype(tmp42)(tmp42 * tmp27);
                            atomic_add(&out_ptr0[static_cast<long>(tmp7 + (479L*tmp3) + (153280L*x3) + (9809920L*x0))], tmp17);
                            atomic_add(&out_ptr1[static_cast<long>(tmp7 + (479L*tmp21) + (153280L*x3) + (9809920L*x0))], tmp28);
                            atomic_add(&out_ptr2[static_cast<long>(tmp32 + (479L*tmp3) + (153280L*x3) + (9809920L*x0))], tmp38);
                            atomic_add(&out_ptr3[static_cast<long>(tmp32 + (479L*tmp21) + (153280L*x3) + (9809920L*x0))], tmp43);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(153280L); x2+=static_cast<long>(8L))
                    {
                        bool tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<bool,8,8>(in_ptr7 + static_cast<long>(x1 + (64L*x2) + (9809920L*x0)), static_cast<long>(64L), tmp0, 8);
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (153280L*x1) + (153280L*x1_inner) + (9809920L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (153280L*x1) + (153280L*x1_inner) + (9809920L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (153280L*x1) + (153280L*x1_inner) + (9809920L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x2 + (153280L*x1) + (153280L*x1_inner) + (9809920L*x0)));
                            auto tmp12 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp1);
                            auto tmp13 = static_cast<float>(1e-05);
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = 1 / std::sqrt(tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            tmp11.store(out_ptr4 + static_cast<long>(x2 + (153280L*x1) + (153280L*x1_inner) + (9809920L*x0)));
                            tmp19.store(tmp20 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp20, 8, out_ptr5 + static_cast<long>(x1 + (64L*x2) + (9809920L*x0)), static_cast<long>(64L));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(153280L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr4 + static_cast<long>(x2 + (153280L*x0) + (9809920L*x1)), static_cast<long>(153280L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr4 + static_cast<long>(x2 + (153280L*x0) + (9809920L*x1)), static_cast<long>(153280L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (64L*x2) + (64L*x2_inner) + (9809920L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(306560L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
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


cpp_fused__unsafe_index_put_add_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_new_zeros_rsub_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const long* in_ptr6,
                       const bool* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr7 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9789440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9789440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9789440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9789440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr3 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(478L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1)];
                            auto tmp4 = in_ptr1[static_cast<long>(x2)];
                            auto tmp14 = in_ptr3[static_cast<long>(x2)];
                            auto tmp16 = in_ptr4[static_cast<long>(x1)];
                            auto tmp18 = in_ptr5[static_cast<long>(x1)];
                            auto tmp29 = in_ptr6[static_cast<long>(x2)];
                            auto tmp1 = decltype(tmp0)(tmp0 + 160);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            auto tmp5 = decltype(tmp4)(tmp4 + 239);
                            auto tmp6 = tmp4 < 0;
                            auto tmp7 = tmp6 ? tmp5 : tmp4;
                            auto tmp8 = c10::convert<long>(x2);
                            auto tmp9 = static_cast<long>(479);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr2[static_cast<long>(128L + x3 + (256L*x2) + (122624L*x1) + (39239680L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp19 = decltype(tmp18)(tmp18 + 160);
                            auto tmp20 = tmp18 < 0;
                            auto tmp21 = tmp20 ? tmp19 : tmp18;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr2[static_cast<long>(128L + x3 + (256L*x2) + (122624L*x1) + (39239680L*x0))];
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp10 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                            auto tmp25 = decltype(tmp24)(tmp24 * tmp14);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp16);
                            auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                            auto tmp30 = decltype(tmp29)(tmp29 + 239);
                            auto tmp31 = tmp29 < 0;
                            auto tmp32 = tmp31 ? tmp30 : tmp29;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr2[static_cast<long>(128L + x3 + (256L*x2) + (122624L*x1) + (39239680L*x0))];
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp10 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp36 = decltype(tmp26)(tmp26 - tmp14);
                            auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                            auto tmp38 = decltype(tmp37)(tmp37 * tmp16);
                            auto tmp39 = [&]
                            {
                                auto tmp40 = in_ptr2[static_cast<long>(128L + x3 + (256L*x2) + (122624L*x1) + (39239680L*x0))];
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp42 = decltype(tmp41)(tmp41 * tmp36);
                            auto tmp43 = decltype(tmp42)(tmp42 * tmp27);
                            atomic_add(&out_ptr0[static_cast<long>(tmp7 + (239L*tmp3) + (38240L*x3) + (4894720L*x0))], tmp17);
                            atomic_add(&out_ptr1[static_cast<long>(tmp7 + (239L*tmp21) + (38240L*x3) + (4894720L*x0))], tmp28);
                            atomic_add(&out_ptr2[static_cast<long>(tmp32 + (239L*tmp3) + (38240L*x3) + (4894720L*x0))], tmp38);
                            atomic_add(&out_ptr3[static_cast<long>(tmp32 + (239L*tmp21) + (38240L*x3) + (4894720L*x0))], tmp43);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38240L); x2+=static_cast<long>(8L))
                    {
                        bool tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<bool,8,8>(in_ptr7 + static_cast<long>(x1 + (128L*x2) + (4894720L*x0)), static_cast<long>(128L), tmp0, 8);
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (38240L*x1) + (38240L*x1_inner) + (4894720L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (38240L*x1) + (38240L*x1_inner) + (4894720L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (38240L*x1) + (38240L*x1_inner) + (4894720L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x2 + (38240L*x1) + (38240L*x1_inner) + (4894720L*x0)));
                            auto tmp12 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp1);
                            auto tmp13 = static_cast<float>(1e-05);
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = 1 / std::sqrt(tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            tmp11.store(out_ptr4 + static_cast<long>(x2 + (38240L*x1) + (38240L*x1_inner) + (4894720L*x0)));
                            tmp19.store(tmp20 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp20, 8, out_ptr5 + static_cast<long>(x1 + (128L*x2) + (4894720L*x0)), static_cast<long>(128L));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(38240L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr4 + static_cast<long>(x2 + (38240L*x0) + (4894720L*x1)), static_cast<long>(38240L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr4 + static_cast<long>(x2 + (38240L*x0) + (4894720L*x1)), static_cast<long>(38240L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (128L*x2) + (128L*x2_inner) + (4894720L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(76480L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
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


cpp_fused__unsafe_index_put_add_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_new_zeros_rsub_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const long* in_ptr6,
                       const bool* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr7 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4874240L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4874240L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4874240L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4874240L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr3 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(238L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1)];
                            auto tmp4 = in_ptr1[static_cast<long>(x2)];
                            auto tmp14 = in_ptr3[static_cast<long>(x2)];
                            auto tmp16 = in_ptr4[static_cast<long>(x1)];
                            auto tmp18 = in_ptr5[static_cast<long>(x1)];
                            auto tmp29 = in_ptr6[static_cast<long>(x2)];
                            auto tmp1 = decltype(tmp0)(tmp0 + 80);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            auto tmp5 = decltype(tmp4)(tmp4 + 119);
                            auto tmp6 = tmp4 < 0;
                            auto tmp7 = tmp6 ? tmp5 : tmp4;
                            auto tmp8 = c10::convert<long>(x2);
                            auto tmp9 = static_cast<long>(239);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr2[static_cast<long>(256L + x3 + (512L*x2) + (122368L*x1) + (19578880L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp19 = decltype(tmp18)(tmp18 + 80);
                            auto tmp20 = tmp18 < 0;
                            auto tmp21 = tmp20 ? tmp19 : tmp18;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr2[static_cast<long>(256L + x3 + (512L*x2) + (122368L*x1) + (19578880L*x0))];
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp10 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                            auto tmp25 = decltype(tmp24)(tmp24 * tmp14);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp16);
                            auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                            auto tmp30 = decltype(tmp29)(tmp29 + 119);
                            auto tmp31 = tmp29 < 0;
                            auto tmp32 = tmp31 ? tmp30 : tmp29;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr2[static_cast<long>(256L + x3 + (512L*x2) + (122368L*x1) + (19578880L*x0))];
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp10 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp36 = decltype(tmp26)(tmp26 - tmp14);
                            auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                            auto tmp38 = decltype(tmp37)(tmp37 * tmp16);
                            auto tmp39 = [&]
                            {
                                auto tmp40 = in_ptr2[static_cast<long>(256L + x3 + (512L*x2) + (122368L*x1) + (19578880L*x0))];
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp42 = decltype(tmp41)(tmp41 * tmp36);
                            auto tmp43 = decltype(tmp42)(tmp42 * tmp27);
                            atomic_add(&out_ptr0[static_cast<long>(tmp7 + (119L*tmp3) + (9520L*x3) + (2437120L*x0))], tmp17);
                            atomic_add(&out_ptr1[static_cast<long>(tmp7 + (119L*tmp21) + (9520L*x3) + (2437120L*x0))], tmp28);
                            atomic_add(&out_ptr2[static_cast<long>(tmp32 + (119L*tmp3) + (9520L*x3) + (2437120L*x0))], tmp38);
                            atomic_add(&out_ptr3[static_cast<long>(tmp32 + (119L*tmp21) + (9520L*x3) + (2437120L*x0))], tmp43);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9520L); x2+=static_cast<long>(8L))
                    {
                        bool tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<bool,8,8>(in_ptr7 + static_cast<long>(x1 + (256L*x2) + (2437120L*x0)), static_cast<long>(256L), tmp0, 8);
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9520L*x1) + (9520L*x1_inner) + (2437120L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (9520L*x1) + (9520L*x1_inner) + (2437120L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (9520L*x1) + (9520L*x1_inner) + (2437120L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x2 + (9520L*x1) + (9520L*x1_inner) + (2437120L*x0)));
                            auto tmp12 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp1);
                            auto tmp13 = static_cast<float>(1e-05);
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = 1 / std::sqrt(tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            tmp11.store(out_ptr4 + static_cast<long>(x2 + (9520L*x1) + (9520L*x1_inner) + (2437120L*x0)));
                            tmp19.store(tmp20 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp20, 8, out_ptr5 + static_cast<long>(x1 + (256L*x2) + (2437120L*x0)), static_cast<long>(256L));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9520L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr4 + static_cast<long>(x2 + (9520L*x0) + (2437120L*x1)), static_cast<long>(9520L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr4 + static_cast<long>(x2 + (9520L*x0) + (2437120L*x1)), static_cast<long>(9520L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (256L*x2) + (256L*x2_inner) + (2437120L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(19040L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
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


cpp_fused__unsafe_index_put_add_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_new_zeros_rsub_threshold_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const long* in_ptr6,
                       const bool* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr7 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2416640L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2416640L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2416640L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2416640L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr3 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(118L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1)];
                            auto tmp4 = in_ptr1[static_cast<long>(x2)];
                            auto tmp14 = in_ptr3[static_cast<long>(x2)];
                            auto tmp16 = in_ptr4[static_cast<long>(x1)];
                            auto tmp18 = in_ptr5[static_cast<long>(x1)];
                            auto tmp29 = in_ptr6[static_cast<long>(x2)];
                            auto tmp1 = decltype(tmp0)(tmp0 + 40);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            auto tmp5 = decltype(tmp4)(tmp4 + 59);
                            auto tmp6 = tmp4 < 0;
                            auto tmp7 = tmp6 ? tmp5 : tmp4;
                            auto tmp8 = c10::convert<long>(x2);
                            auto tmp9 = static_cast<long>(119);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr2[static_cast<long>(512L + x3 + (1024L*x2) + (121856L*x1) + (9748480L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp19 = decltype(tmp18)(tmp18 + 40);
                            auto tmp20 = tmp18 < 0;
                            auto tmp21 = tmp20 ? tmp19 : tmp18;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr2[static_cast<long>(512L + x3 + (1024L*x2) + (121856L*x1) + (9748480L*x0))];
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp10 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                            auto tmp25 = decltype(tmp24)(tmp24 * tmp14);
                            auto tmp26 = static_cast<float>(1.0);
                            auto tmp27 = decltype(tmp26)(tmp26 - tmp16);
                            auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                            auto tmp30 = decltype(tmp29)(tmp29 + 59);
                            auto tmp31 = tmp29 < 0;
                            auto tmp32 = tmp31 ? tmp30 : tmp29;
                            auto tmp33 = [&]
                            {
                                auto tmp34 = in_ptr2[static_cast<long>(512L + x3 + (1024L*x2) + (121856L*x1) + (9748480L*x0))];
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp10 ? tmp33() : static_cast<decltype(tmp33())>(0.0);
                            auto tmp36 = decltype(tmp26)(tmp26 - tmp14);
                            auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                            auto tmp38 = decltype(tmp37)(tmp37 * tmp16);
                            auto tmp39 = [&]
                            {
                                auto tmp40 = in_ptr2[static_cast<long>(512L + x3 + (1024L*x2) + (121856L*x1) + (9748480L*x0))];
                                return tmp40;
                            }
                            ;
                            auto tmp41 = tmp10 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp42 = decltype(tmp41)(tmp41 * tmp36);
                            auto tmp43 = decltype(tmp42)(tmp42 * tmp27);
                            atomic_add(&out_ptr0[static_cast<long>(tmp7 + (59L*tmp3) + (2360L*x3) + (1208320L*x0))], tmp17);
                            atomic_add(&out_ptr2[static_cast<long>(tmp7 + (59L*tmp21) + (2360L*x3) + (1208320L*x0))], tmp28);
                            atomic_add(&out_ptr3[static_cast<long>(tmp32 + (59L*tmp3) + (2360L*x3) + (1208320L*x0))], tmp38);
                            atomic_add(&out_ptr1[static_cast<long>(tmp32 + (59L*tmp21) + (2360L*x3) + (1208320L*x0))], tmp43);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2360L); x2+=static_cast<long>(8L))
                    {
                        bool tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<bool,8,8>(in_ptr7 + static_cast<long>(x1 + (512L*x2) + (1208320L*x0)), static_cast<long>(512L), tmp0, 8);
                        float tmp20[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (2360L*x1) + (2360L*x1_inner) + (1208320L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (2360L*x1) + (2360L*x1_inner) + (1208320L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x2 + (2360L*x1) + (2360L*x1_inner) + (1208320L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (2360L*x1) + (2360L*x1_inner) + (1208320L*x0)));
                            auto tmp12 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp16 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp1);
                            auto tmp13 = static_cast<float>(1e-05);
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = 1 / std::sqrt(tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp11 * tmp18;
                            tmp11.store(out_ptr4 + static_cast<long>(x2 + (2360L*x1) + (2360L*x1_inner) + (1208320L*x0)));
                            tmp19.store(tmp20 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp20, 8, out_ptr5 + static_cast<long>(x1 + (512L*x2) + (1208320L*x0)), static_cast<long>(512L));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(2360L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr4 + static_cast<long>(x2 + (2360L*x0) + (1208320L*x1)), static_cast<long>(2360L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr4 + static_cast<long>(x2 + (2360L*x0) + (1208320L*x1)), static_cast<long>(2360L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (512L*x2) + (512L*x2_inner) + (1208320L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4720L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4720L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(19040L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(19040L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(76480L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(76480L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(306560L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(306560L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1227520L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1227520L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, convolution, relu, convolution_1, relu_1, getitem, getitem_1, convolution_2, relu_2, convolution_3, relu_3, getitem_2, getitem_3, convolution_4, relu_4, convolution_5, relu_5, getitem_4, getitem_5, convolution_6, relu_6, convolution_7, relu_7, getitem_6, getitem_7, convolution_8, relu_8, convolution_9, convert_element_type_26, convert_element_type_27, unsqueeze_81, unsqueeze_82, sub_10, sub_12, cat, convolution_10, relu_10, convolution_11, convert_element_type_38, convert_element_type_39, unsqueeze_100, unsqueeze_101, sub_16, sub_18, cat_1, convolution_12, relu_12, convolution_13, convert_element_type_50, convert_element_type_51, unsqueeze_119, unsqueeze_120, sub_22, sub_24, cat_2, convolution_14, relu_14, convolution_15, convert_element_type_62, convert_element_type_63, unsqueeze_138, unsqueeze_139, sub_28, sub_30, cat_3, convolution_16, relu_16, convolution_17, relu_17, le_2, le_4, le_6, le_8, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_9, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_17, (256, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_21, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (512, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_29, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_33, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_37, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_41, (512, 1024, 3, 3), (9216, 1, 3072, 1024))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_45, (256, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (256, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_53, (128, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_57, (128, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_61, (64, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_65, (64, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_69, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_73, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_81, (128, ), (1, ))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_117, (128, ), (1, ))
    assert_size_stride(primals_118, (128, ), (1, ))
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (64, ), (1, ))
    assert_size_stride(primals_126, (64, ), (1, ))
    assert_size_stride(primals_127, (64, ), (1, ))
    assert_size_stride(primals_129, (2, 3, 640, 959), (1841280, 1, 2877, 3))
    assert_size_stride(convolution, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    assert_size_stride(relu, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    assert_size_stride(convolution_1, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    assert_size_stride(relu_1, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    assert_size_stride(getitem, (2, 64, 320, 479), (9809920, 1, 30656, 64))
    assert_size_stride(getitem_1, (2, 64, 320, 479), (9809920, 1, 30656, 64))
    assert_size_stride(convolution_2, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    assert_size_stride(relu_2, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    assert_size_stride(convolution_3, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    assert_size_stride(relu_3, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    assert_size_stride(getitem_2, (2, 128, 160, 239), (4894720, 1, 30592, 128))
    assert_size_stride(getitem_3, (2, 128, 160, 239), (4894720, 1, 30592, 128))
    assert_size_stride(convolution_4, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    assert_size_stride(relu_4, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    assert_size_stride(convolution_5, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    assert_size_stride(relu_5, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    assert_size_stride(getitem_4, (2, 256, 80, 119), (2437120, 1, 30464, 256))
    assert_size_stride(getitem_5, (2, 256, 80, 119), (2437120, 1, 30464, 256))
    assert_size_stride(convolution_6, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    assert_size_stride(relu_6, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    assert_size_stride(convolution_7, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    assert_size_stride(relu_7, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    assert_size_stride(getitem_6, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    assert_size_stride(getitem_7, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    assert_size_stride(convolution_8, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    assert_size_stride(relu_8, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    assert_size_stride(convolution_9, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    assert_size_stride(convert_element_type_26, (118, ), (1, ))
    assert_size_stride(convert_element_type_27, (118, ), (1, ))
    assert_size_stride(unsqueeze_81, (80, 1), (1, 1))
    assert_size_stride(unsqueeze_82, (80, 1), (1, 1))
    assert_size_stride(sub_10, (80, 1), (1, 1))
    assert_size_stride(sub_12, (118, ), (1, ))
    assert_size_stride(cat, (2, 1024, 80, 119), (9748480, 1, 121856, 1024))
    assert_size_stride(convolution_10, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    assert_size_stride(relu_10, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    assert_size_stride(convolution_11, (2, 256, 80, 119), (2437120, 1, 30464, 256))
    assert_size_stride(convert_element_type_38, (238, ), (1, ))
    assert_size_stride(convert_element_type_39, (238, ), (1, ))
    assert_size_stride(unsqueeze_100, (160, 1), (1, 1))
    assert_size_stride(unsqueeze_101, (160, 1), (1, 1))
    assert_size_stride(sub_16, (160, 1), (1, 1))
    assert_size_stride(sub_18, (238, ), (1, ))
    assert_size_stride(cat_1, (2, 512, 160, 239), (19578880, 1, 122368, 512))
    assert_size_stride(convolution_12, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    assert_size_stride(relu_12, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    assert_size_stride(convolution_13, (2, 128, 160, 239), (4894720, 1, 30592, 128))
    assert_size_stride(convert_element_type_50, (478, ), (1, ))
    assert_size_stride(convert_element_type_51, (478, ), (1, ))
    assert_size_stride(unsqueeze_119, (320, 1), (1, 1))
    assert_size_stride(unsqueeze_120, (320, 1), (1, 1))
    assert_size_stride(sub_22, (320, 1), (1, 1))
    assert_size_stride(sub_24, (478, ), (1, ))
    assert_size_stride(cat_2, (2, 256, 320, 479), (39239680, 1, 122624, 256))
    assert_size_stride(convolution_14, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    assert_size_stride(relu_14, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    assert_size_stride(convolution_15, (2, 64, 320, 479), (9809920, 1, 30656, 64))
    assert_size_stride(convert_element_type_62, (958, ), (1, ))
    assert_size_stride(convert_element_type_63, (958, ), (1, ))
    assert_size_stride(unsqueeze_138, (640, 1), (1, 1))
    assert_size_stride(unsqueeze_139, (640, 1), (1, 1))
    assert_size_stride(sub_28, (640, 1), (1, 1))
    assert_size_stride(sub_30, (958, ), (1, ))
    assert_size_stride(cat_3, (2, 128, 640, 959), (78561280, 1, 122752, 128))
    assert_size_stride(convolution_16, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    assert_size_stride(relu_16, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    assert_size_stride(convolution_17, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    assert_size_stride(relu_17, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    assert_size_stride(le_2, (2, 64, 320, 479), (9809920, 1, 30656, 64))
    assert_size_stride(le_4, (2, 128, 160, 239), (4894720, 1, 30592, 128))
    assert_size_stride(le_6, (2, 256, 80, 119), (2437120, 1, 30464, 256))
    assert_size_stride(le_8, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    assert_size_stride(tangents_1, (2, 2, 640, 959), (1227520, 613760, 959, 1))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf0 = aten.convolution_backward(tangents_1, relu_17, primals_73, [2], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del primals_73
    del tangents_1
    buf1 = buf0[0]
    buf2 = buf0[1]
    buf3 = buf0[2]
    del buf0
    buf4 = empty((64, ), device='cpu', dtype=torch.float32)
    buf5 = empty((64, ), device='cpu', dtype=torch.float32)
    buf6 = buf5; del buf5  # reuse
    buf7 = buf1; del buf1  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_0(c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf4.data_ptr()))
    del convolution_17
    del primals_126
    del primals_127
    del primals_71
    del relu_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf8 = aten.convolution_backward(buf7, relu_16, primals_69, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf7
    del primals_69
    buf9 = buf8[0]
    buf10 = buf8[1]
    buf11 = buf8[2]
    del buf8
    buf12 = empty((64, ), device='cpu', dtype=torch.float32)
    buf13 = empty((64, ), device='cpu', dtype=torch.float32)
    buf14 = buf13; del buf13  # reuse
    buf15 = buf9; del buf9  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1(c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf12.data_ptr()))
    del convolution_16
    del primals_123
    del primals_124
    del primals_67
    del relu_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf16 = aten.convolution_backward(buf15, cat_3, primals_65, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf15
    del cat_3
    del primals_65
    buf17 = buf16[0]
    buf18 = buf16[1]
    buf19 = buf16[2]
    del buf16
    buf20 = empty((2, 64, 320, 479), device='cpu', dtype=torch.float32)
    buf22 = empty((2, 64, 320, 479), device='cpu', dtype=torch.float32)
    buf24 = empty((2, 64, 320, 479), device='cpu', dtype=torch.float32)
    buf26 = empty((2, 64, 320, 479), device='cpu', dtype=torch.float32)
    buf28 = empty((2, 64, 320, 479), device='cpu', dtype=torch.float32)
    buf32 = empty_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.float32)
    buf29 = empty((64, ), device='cpu', dtype=torch.float32)
    buf30 = empty((64, ), device='cpu', dtype=torch.float32)
    buf31 = buf30; del buf30  # reuse
    cpp_fused__unsafe_index_put_add_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_new_zeros_rsub_threshold_backward_2(c_void_p(buf31.data_ptr()), c_void_p(unsqueeze_139.data_ptr()), c_void_p(convert_element_type_63.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(sub_30.data_ptr()), c_void_p(sub_28.data_ptr()), c_void_p(unsqueeze_138.data_ptr()), c_void_p(convert_element_type_62.data_ptr()), c_void_p(le_2.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf29.data_ptr()))
    del buf20
    del buf22
    del buf24
    del buf26
    del buf28
    del convert_element_type_62
    del convert_element_type_63
    del convolution_15
    del le_2
    del primals_120
    del primals_121
    del primals_63
    del sub_28
    del sub_30
    del unsqueeze_138
    del unsqueeze_139
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf33 = aten.convolution_backward(buf32, relu_14, primals_61, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf32
    del primals_61
    buf34 = buf33[0]
    buf35 = buf33[1]
    buf36 = buf33[2]
    del buf33
    buf37 = empty((128, ), device='cpu', dtype=torch.float32)
    buf38 = empty((128, ), device='cpu', dtype=torch.float32)
    buf39 = buf38; del buf38  # reuse
    buf40 = buf34; del buf34  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3(c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf37.data_ptr()))
    del convolution_14
    del primals_117
    del primals_118
    del primals_59
    del relu_14
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf41 = aten.convolution_backward(buf40, cat_2, primals_57, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf40
    del cat_2
    del primals_57
    buf42 = buf41[0]
    buf43 = buf41[1]
    buf44 = buf41[2]
    del buf41
    buf45 = empty((2, 128, 160, 239), device='cpu', dtype=torch.float32)
    buf47 = empty((2, 128, 160, 239), device='cpu', dtype=torch.float32)
    buf49 = empty((2, 128, 160, 239), device='cpu', dtype=torch.float32)
    buf51 = empty((2, 128, 160, 239), device='cpu', dtype=torch.float32)
    buf53 = empty((2, 128, 160, 239), device='cpu', dtype=torch.float32)
    buf57 = empty_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.float32)
    buf54 = empty((128, ), device='cpu', dtype=torch.float32)
    buf55 = empty((128, ), device='cpu', dtype=torch.float32)
    buf56 = buf55; del buf55  # reuse
    cpp_fused__unsafe_index_put_add_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_new_zeros_rsub_threshold_backward_4(c_void_p(buf56.data_ptr()), c_void_p(unsqueeze_120.data_ptr()), c_void_p(convert_element_type_51.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(sub_24.data_ptr()), c_void_p(sub_22.data_ptr()), c_void_p(unsqueeze_119.data_ptr()), c_void_p(convert_element_type_50.data_ptr()), c_void_p(le_4.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf54.data_ptr()))
    del buf45
    del buf47
    del buf49
    del buf51
    del buf53
    del convert_element_type_50
    del convert_element_type_51
    del convolution_13
    del le_4
    del primals_114
    del primals_115
    del primals_55
    del sub_22
    del sub_24
    del unsqueeze_119
    del unsqueeze_120
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf58 = aten.convolution_backward(buf57, relu_12, primals_53, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf57
    del primals_53
    buf59 = buf58[0]
    buf60 = buf58[1]
    buf61 = buf58[2]
    del buf58
    buf62 = empty((256, ), device='cpu', dtype=torch.float32)
    buf63 = empty((256, ), device='cpu', dtype=torch.float32)
    buf64 = buf63; del buf63  # reuse
    buf65 = buf59; del buf59  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf62.data_ptr()))
    del convolution_12
    del primals_111
    del primals_112
    del primals_51
    del relu_12
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf66 = aten.convolution_backward(buf65, cat_1, primals_49, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf65
    del cat_1
    del primals_49
    buf67 = buf66[0]
    buf68 = buf66[1]
    buf69 = buf66[2]
    del buf66
    buf70 = empty((2, 256, 80, 119), device='cpu', dtype=torch.float32)
    buf72 = empty((2, 256, 80, 119), device='cpu', dtype=torch.float32)
    buf74 = empty((2, 256, 80, 119), device='cpu', dtype=torch.float32)
    buf76 = empty((2, 256, 80, 119), device='cpu', dtype=torch.float32)
    buf78 = empty((2, 256, 80, 119), device='cpu', dtype=torch.float32)
    buf82 = empty_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.float32)
    buf79 = empty((256, ), device='cpu', dtype=torch.float32)
    buf80 = empty((256, ), device='cpu', dtype=torch.float32)
    buf81 = buf80; del buf80  # reuse
    cpp_fused__unsafe_index_put_add_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_new_zeros_rsub_threshold_backward_6(c_void_p(buf81.data_ptr()), c_void_p(unsqueeze_101.data_ptr()), c_void_p(convert_element_type_39.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(sub_18.data_ptr()), c_void_p(sub_16.data_ptr()), c_void_p(unsqueeze_100.data_ptr()), c_void_p(convert_element_type_38.data_ptr()), c_void_p(le_6.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf79.data_ptr()))
    del buf70
    del buf72
    del buf74
    del buf76
    del buf78
    del convert_element_type_38
    del convert_element_type_39
    del convolution_11
    del le_6
    del primals_108
    del primals_109
    del primals_47
    del sub_16
    del sub_18
    del unsqueeze_100
    del unsqueeze_101
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf83 = aten.convolution_backward(buf82, relu_10, primals_45, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf82
    del primals_45
    buf84 = buf83[0]
    buf85 = buf83[1]
    buf86 = buf83[2]
    del buf83
    buf87 = empty((512, ), device='cpu', dtype=torch.float32)
    buf88 = empty((512, ), device='cpu', dtype=torch.float32)
    buf89 = buf88; del buf88  # reuse
    buf90 = buf84; del buf84  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7(c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf87.data_ptr()))
    del convolution_10
    del primals_105
    del primals_106
    del primals_43
    del relu_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf91 = aten.convolution_backward(buf90, cat, primals_41, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf90
    del cat
    del primals_41
    buf92 = buf91[0]
    buf93 = buf91[1]
    buf94 = buf91[2]
    del buf91
    buf95 = empty((2, 512, 40, 59), device='cpu', dtype=torch.float32)
    buf101 = empty((2, 512, 40, 59), device='cpu', dtype=torch.float32)
    buf97 = empty((2, 512, 40, 59), device='cpu', dtype=torch.float32)
    buf99 = empty((2, 512, 40, 59), device='cpu', dtype=torch.float32)
    buf103 = empty((2, 512, 40, 59), device='cpu', dtype=torch.float32)
    buf107 = empty_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.float32)
    buf104 = empty((512, ), device='cpu', dtype=torch.float32)
    buf105 = empty((512, ), device='cpu', dtype=torch.float32)
    buf106 = buf105; del buf105  # reuse
    cpp_fused__unsafe_index_put_add_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_new_zeros_rsub_threshold_backward_8(c_void_p(buf106.data_ptr()), c_void_p(unsqueeze_82.data_ptr()), c_void_p(convert_element_type_27.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(sub_12.data_ptr()), c_void_p(sub_10.data_ptr()), c_void_p(unsqueeze_81.data_ptr()), c_void_p(convert_element_type_26.data_ptr()), c_void_p(le_8.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf104.data_ptr()))
    del buf101
    del buf103
    del buf95
    del buf97
    del buf99
    del convert_element_type_26
    del convert_element_type_27
    del convolution_9
    del le_8
    del primals_102
    del primals_103
    del primals_39
    del sub_10
    del sub_12
    del unsqueeze_81
    del unsqueeze_82
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf108 = aten.convolution_backward(buf107, relu_8, primals_37, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf107
    del primals_37
    buf109 = buf108[0]
    buf110 = buf108[1]
    buf111 = buf108[2]
    del buf108
    buf112 = empty((512, ), device='cpu', dtype=torch.float32)
    buf113 = empty((512, ), device='cpu', dtype=torch.float32)
    buf114 = buf113; del buf113  # reuse
    buf115 = buf109; del buf109  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf112.data_ptr()))
    del convolution_8
    del primals_100
    del primals_35
    del primals_99
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf116 = aten.convolution_backward(buf115, getitem_6, primals_33, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf115
    del getitem_6
    del primals_33
    buf117 = buf116[0]
    buf118 = buf116[1]
    buf119 = buf116[2]
    del buf116
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf120 = aten.max_pool2d_with_indices_backward(buf117, relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_7)
    del buf117
    del getitem_7
    buf121 = buf120
    del buf120
    buf122 = empty((512, ), device='cpu', dtype=torch.float32)
    buf123 = empty((512, ), device='cpu', dtype=torch.float32)
    buf124 = buf123; del buf123  # reuse
    buf125 = buf121; del buf121  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf122.data_ptr()))
    del buf92
    del convolution_7
    del primals_31
    del primals_96
    del primals_97
    del relu_7
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf126 = aten.convolution_backward(buf125, relu_6, primals_29, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf125
    del primals_29
    buf127 = buf126[0]
    buf128 = buf126[1]
    buf129 = buf126[2]
    del buf126
    buf130 = empty((512, ), device='cpu', dtype=torch.float32)
    buf131 = empty((512, ), device='cpu', dtype=torch.float32)
    buf132 = buf131; del buf131  # reuse
    buf133 = buf127; del buf127  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf130.data_ptr()))
    del convolution_6
    del primals_27
    del primals_93
    del primals_94
    del relu_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf134 = aten.convolution_backward(buf133, getitem_4, primals_25, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf133
    del getitem_4
    del primals_25
    buf135 = buf134[0]
    buf136 = buf134[1]
    buf137 = buf134[2]
    del buf134
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf138 = aten.max_pool2d_with_indices_backward(buf135, relu_5, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_5)
    del buf135
    del getitem_5
    buf139 = buf138
    del buf138
    buf140 = empty((256, ), device='cpu', dtype=torch.float32)
    buf141 = empty((256, ), device='cpu', dtype=torch.float32)
    buf142 = buf141; del buf141  # reuse
    buf143 = buf139; del buf139  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf140.data_ptr()))
    del buf67
    del convolution_5
    del primals_23
    del primals_90
    del primals_91
    del relu_5
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf144 = aten.convolution_backward(buf143, relu_4, primals_21, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf143
    del primals_21
    buf145 = buf144[0]
    buf146 = buf144[1]
    buf147 = buf144[2]
    del buf144
    buf148 = empty((256, ), device='cpu', dtype=torch.float32)
    buf149 = empty((256, ), device='cpu', dtype=torch.float32)
    buf150 = buf149; del buf149  # reuse
    buf151 = buf145; del buf145  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf148.data_ptr()))
    del convolution_4
    del primals_19
    del primals_87
    del primals_88
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf152 = aten.convolution_backward(buf151, getitem_2, primals_17, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf151
    del getitem_2
    del primals_17
    buf153 = buf152[0]
    buf154 = buf152[1]
    buf155 = buf152[2]
    del buf152
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf156 = aten.max_pool2d_with_indices_backward(buf153, relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_3)
    del buf153
    del getitem_3
    buf157 = buf156
    del buf156
    buf158 = empty((128, ), device='cpu', dtype=torch.float32)
    buf159 = empty((128, ), device='cpu', dtype=torch.float32)
    buf160 = buf159; del buf159  # reuse
    buf161 = buf157; del buf157  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf158.data_ptr()))
    del buf42
    del convolution_3
    del primals_15
    del primals_84
    del primals_85
    del relu_3
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf162 = aten.convolution_backward(buf161, relu_2, primals_13, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf161
    del primals_13
    buf163 = buf162[0]
    buf164 = buf162[1]
    buf165 = buf162[2]
    del buf162
    buf166 = empty((128, ), device='cpu', dtype=torch.float32)
    buf167 = empty((128, ), device='cpu', dtype=torch.float32)
    buf168 = buf167; del buf167  # reuse
    buf169 = buf163; del buf163  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf166.data_ptr()))
    del convolution_2
    del primals_11
    del primals_81
    del primals_82
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf170 = aten.convolution_backward(buf169, getitem, primals_9, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf169
    del getitem
    del primals_9
    buf171 = buf170[0]
    buf172 = buf170[1]
    buf173 = buf170[2]
    del buf170
    # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
    buf174 = aten.max_pool2d_with_indices_backward(buf171, relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_1)
    del buf171
    del getitem_1
    buf175 = buf174
    del buf174
    buf176 = empty((64, ), device='cpu', dtype=torch.float32)
    buf177 = empty((64, ), device='cpu', dtype=torch.float32)
    buf178 = buf177; del buf177  # reuse
    buf179 = buf175; del buf175  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_16(c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf176.data_ptr()))
    del buf17
    del convolution_1
    del primals_7
    del primals_78
    del primals_79
    del relu_1
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf180 = aten.convolution_backward(buf179, relu, primals_5, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf179
    del primals_5
    buf181 = buf180[0]
    buf182 = buf180[1]
    buf183 = buf180[2]
    del buf180
    buf184 = empty((64, ), device='cpu', dtype=torch.float32)
    buf185 = empty((64, ), device='cpu', dtype=torch.float32)
    buf186 = buf185; del buf185  # reuse
    buf187 = buf181; del buf181  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf184.data_ptr()))
    del convolution
    del primals_3
    del primals_75
    del primals_76
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf188 = aten.convolution_backward(buf187, primals_129, primals_1, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf187
    del primals_1
    del primals_129
    buf189 = buf188[1]
    buf190 = buf188[2]
    return (buf189, buf190, buf186, buf184, buf182, buf183, buf178, buf176, buf172, buf173, buf168, buf166, buf164, buf165, buf160, buf158, buf154, buf155, buf150, buf148, buf146, buf147, buf142, buf140, buf136, buf137, buf132, buf130, buf128, buf129, buf124, buf122, buf118, buf119, buf114, buf112, buf110, buf111, buf106, buf104, buf93, buf94, buf89, buf87, buf85, buf86, buf81, buf79, buf68, buf69, buf64, buf62, buf60, buf61, buf56, buf54, buf43, buf44, buf39, buf37, buf35, buf36, buf31, buf29, buf18, buf19, buf14, buf12, buf10, buf11, buf6, buf4, buf2, buf3, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, 1024, 3, 3), (9216, 1, 3072, 1024), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((256, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((256, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((128, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((64, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((64, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((2, 3, 640, 959), (1841280, 1, 2877, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    relu = rand_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    getitem = rand_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.int64)
    convolution_2 = rand_strided((2, 128, 320, 479), (19619840, 1, 61312, 128), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((2, 128, 320, 479), (19619840, 1, 61312, 128), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((2, 128, 320, 479), (19619840, 1, 61312, 128), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((2, 128, 320, 479), (19619840, 1, 61312, 128), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.int64)
    convolution_4 = rand_strided((2, 256, 160, 239), (9789440, 1, 61184, 256), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((2, 256, 160, 239), (9789440, 1, 61184, 256), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((2, 256, 160, 239), (9789440, 1, 61184, 256), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((2, 256, 160, 239), (9789440, 1, 61184, 256), device='cpu', dtype=torch.float32)
    getitem_4 = rand_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.int64)
    convolution_6 = rand_strided((2, 512, 80, 119), (4874240, 1, 60928, 512), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((2, 512, 80, 119), (4874240, 1, 60928, 512), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((2, 512, 80, 119), (4874240, 1, 60928, 512), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((2, 512, 80, 119), (4874240, 1, 60928, 512), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.int64)
    convolution_8 = rand_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.float32)
    convert_element_type_26 = rand_strided((118, ), (1, ), device='cpu', dtype=torch.int64)
    convert_element_type_27 = rand_strided((118, ), (1, ), device='cpu', dtype=torch.int64)
    unsqueeze_81 = rand_strided((80, 1), (1, 1), device='cpu', dtype=torch.int64)
    unsqueeze_82 = rand_strided((80, 1), (1, 1), device='cpu', dtype=torch.int64)
    sub_10 = rand_strided((80, 1), (1, 1), device='cpu', dtype=torch.float32)
    sub_12 = rand_strided((118, ), (1, ), device='cpu', dtype=torch.float32)
    cat = rand_strided((2, 1024, 80, 119), (9748480, 1, 121856, 1024), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((2, 512, 80, 119), (4874240, 1, 60928, 512), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((2, 512, 80, 119), (4874240, 1, 60928, 512), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.float32)
    convert_element_type_38 = rand_strided((238, ), (1, ), device='cpu', dtype=torch.int64)
    convert_element_type_39 = rand_strided((238, ), (1, ), device='cpu', dtype=torch.int64)
    unsqueeze_100 = rand_strided((160, 1), (1, 1), device='cpu', dtype=torch.int64)
    unsqueeze_101 = rand_strided((160, 1), (1, 1), device='cpu', dtype=torch.int64)
    sub_16 = rand_strided((160, 1), (1, 1), device='cpu', dtype=torch.float32)
    sub_18 = rand_strided((238, ), (1, ), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((2, 512, 160, 239), (19578880, 1, 122368, 512), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((2, 256, 160, 239), (9789440, 1, 61184, 256), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((2, 256, 160, 239), (9789440, 1, 61184, 256), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.float32)
    convert_element_type_50 = rand_strided((478, ), (1, ), device='cpu', dtype=torch.int64)
    convert_element_type_51 = rand_strided((478, ), (1, ), device='cpu', dtype=torch.int64)
    unsqueeze_119 = rand_strided((320, 1), (1, 1), device='cpu', dtype=torch.int64)
    unsqueeze_120 = rand_strided((320, 1), (1, 1), device='cpu', dtype=torch.int64)
    sub_22 = rand_strided((320, 1), (1, 1), device='cpu', dtype=torch.float32)
    sub_24 = rand_strided((478, ), (1, ), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((2, 256, 320, 479), (39239680, 1, 122624, 256), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((2, 128, 320, 479), (19619840, 1, 61312, 128), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((2, 128, 320, 479), (19619840, 1, 61312, 128), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.float32)
    convert_element_type_62 = rand_strided((958, ), (1, ), device='cpu', dtype=torch.int64)
    convert_element_type_63 = rand_strided((958, ), (1, ), device='cpu', dtype=torch.int64)
    unsqueeze_138 = rand_strided((640, 1), (1, 1), device='cpu', dtype=torch.int64)
    unsqueeze_139 = rand_strided((640, 1), (1, 1), device='cpu', dtype=torch.int64)
    sub_28 = rand_strided((640, 1), (1, 1), device='cpu', dtype=torch.float32)
    sub_30 = rand_strided((958, ), (1, ), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((2, 128, 640, 959), (78561280, 1, 122752, 128), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((2, 64, 640, 959), (39280640, 1, 61376, 64), device='cpu', dtype=torch.float32)
    le_2 = rand_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.bool)
    le_4 = rand_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.bool)
    le_6 = rand_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.bool)
    le_8 = rand_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((2, 2, 640, 959), (1227520, 613760, 959, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, convolution, relu, convolution_1, relu_1, getitem, getitem_1, convolution_2, relu_2, convolution_3, relu_3, getitem_2, getitem_3, convolution_4, relu_4, convolution_5, relu_5, getitem_4, getitem_5, convolution_6, relu_6, convolution_7, relu_7, getitem_6, getitem_7, convolution_8, relu_8, convolution_9, convert_element_type_26, convert_element_type_27, unsqueeze_81, unsqueeze_82, sub_10, sub_12, cat, convolution_10, relu_10, convolution_11, convert_element_type_38, convert_element_type_39, unsqueeze_100, unsqueeze_101, sub_16, sub_18, cat_1, convolution_12, relu_12, convolution_13, convert_element_type_50, convert_element_type_51, unsqueeze_119, unsqueeze_120, sub_22, sub_24, cat_2, convolution_14, relu_14, convolution_15, convert_element_type_62, convert_element_type_63, unsqueeze_138, unsqueeze_139, sub_28, sub_30, cat_3, convolution_16, relu_16, convolution_17, relu_17, le_2, le_4, le_6, le_8, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pytorch_unet', benchmark_compiled_module)
