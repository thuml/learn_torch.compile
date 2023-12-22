
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (1536L*x2) + (75264L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1536L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1536L*x2) + (75264L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
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
                        tmp25.store(out_ptr4 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(264L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (264L*x1)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(264L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (264L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (264L*x0)));
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
                tmp18.store(out_ptr2 + static_cast<long>(x1 + (264L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(264L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (1584L*x2) + (77616L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(792);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (38808L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(1584);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-38808L) + x2 + (49L*x1) + (38808L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (1584L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1056L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_4 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (1584L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (1584L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (1584L*x2) + (77616L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(792);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (38808L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(1584);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-38808L) + x2 + (49L*x1) + (38808L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(49.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (77616L*x0))] = tmp29;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (1584L*x2_inner) + (77616L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (77616L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (77616L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.002551020408163265);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_5 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr4[static_cast<long>(x1 + (1584L*x2) + (77616L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(396);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (19404L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(792);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-19404L) + x2 + (49L*x1) + (19404L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(1188);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = tmp15 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr2[static_cast<long>((-38808L) + x2 + (49L*x1) + (19404L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp0 >= tmp16;
                        auto tmp23 = static_cast<long>(1584);
                        auto tmp24 = tmp0 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-58212L) + x2 + (49L*x1) + (19404L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        auto tmp29 = tmp11 ? tmp14 : tmp28;
                        auto tmp30 = tmp4 ? tmp7 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (77616L*x0))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (1584L*x2_inner) + (77616L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (77616L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (77616L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.002551020408163265);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(264L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (264L*x1)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(264L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (264L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (264L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (264L*x0)));
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
                tmp20.store(out_ptr2 + static_cast<long>(x1 + (264L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(264L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (1584L*x2) + (77616L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(792);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (38808L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(1584);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-38808L) + x2 + (49L*x1) + (38808L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (1584L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1056L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_9 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (1584L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (1584L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (1584L*x2) + (77616L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(792);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (38808L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(1584);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-38808L) + x2 + (49L*x1) + (38808L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(49.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (77616L*x0))] = tmp29;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (1584L*x2_inner) + (77616L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (77616L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (77616L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.002551020408163265);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_10 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr4[static_cast<long>(x1 + (1584L*x2) + (77616L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(396);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (19404L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(792);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-19404L) + x2 + (49L*x1) + (19404L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(1188);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = tmp15 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr2[static_cast<long>((-38808L) + x2 + (49L*x1) + (19404L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp0 >= tmp16;
                        auto tmp23 = static_cast<long>(1584);
                        auto tmp24 = tmp0 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-58212L) + x2 + (49L*x1) + (19404L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        auto tmp29 = tmp11 ? tmp14 : tmp28;
                        auto tmp30 = tmp4 ? tmp7 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (77616L*x0))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (1584L*x2_inner) + (77616L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (77616L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (77616L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.002551020408163265);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_11 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(264L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (264L*x1)));
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(264L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (264L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (264L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (264L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (264L*x0)));
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
                tmp22.store(out_ptr2 + static_cast<long>(x1 + (264L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(264L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (1584L*x2) + (77616L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(792);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (38808L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(1584);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-38808L) + x2 + (49L*x1) + (38808L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (1584L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1056L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (1584L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (1584L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (1584L*x2) + (77616L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(792);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (38808L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(1584);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-38808L) + x2 + (49L*x1) + (38808L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(49.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (77616L*x0))] = tmp29;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (1584L*x2_inner) + (77616L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (77616L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (77616L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.002551020408163265);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_15 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr4[static_cast<long>(x1 + (1584L*x2) + (77616L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(396);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (19404L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(792);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-19404L) + x2 + (49L*x1) + (19404L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(1188);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = tmp15 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr2[static_cast<long>((-38808L) + x2 + (49L*x1) + (19404L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp0 >= tmp16;
                        auto tmp23 = static_cast<long>(1584);
                        auto tmp24 = tmp0 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-58212L) + x2 + (49L*x1) + (19404L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        auto tmp29 = tmp11 ? tmp14 : tmp28;
                        auto tmp30 = tmp4 ? tmp7 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (77616L*x0))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (49L*x0) + (77616L*x1)), static_cast<long>(49L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (1584L*x2_inner) + (77616L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (77616L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1584L*x2) + (77616L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.002551020408163265);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(264L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (264L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (264L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(264L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (264L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (264L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (264L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (264L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (264L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
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
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (264L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(264L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_19 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(49.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp10 = decltype(tmp9)(1)/(decltype(tmp9)(1) + tmp9.neg().exp());
                            auto tmp11 = static_cast<float>(1.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp12 - tmp10;
                            auto tmp14 = tmp9 * tmp13;
                            auto tmp15 = tmp14 + tmp12;
                            auto tmp16 = tmp10 * tmp15;
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp20 = tmp18 - tmp19;
                            auto tmp21 = tmp17 * tmp20;
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                            tmp_acc1_vec = tmp_acc1_vec + tmp21;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(49.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp3 + tmp7;
                        auto tmp10 = decltype(tmp9)(1)/(decltype(tmp9)(1) + tmp9.neg().exp());
                        auto tmp11 = static_cast<float>(1.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp12 - tmp10;
                        auto tmp14 = tmp9 * tmp13;
                        auto tmp15 = tmp14 + tmp12;
                        auto tmp16 = tmp10 * tmp15;
                        auto tmp17 = tmp8 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp22 = static_cast<float>(0.002551020408163265);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(720L + x1 + (960L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(720L + x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(720L + x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp4.store(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(480L + x1 + (960L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(480L + x1));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(480L + x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp4 = tmp0 * tmp3;
                tmp4.store(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(240L + x1 + (960L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(240L + x1));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(240L + x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp4 = tmp0 * tmp3;
                tmp4.store(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp4 = tmp0 * tmp3;
                tmp4.store(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_23 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(960L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(240);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x1);
                                auto tmp7 = static_cast<long>(15);
                                auto tmp8 = tmp6 < tmp7;
                                auto tmp9 = c10::convert<long>(x2);
                                auto tmp10 = tmp9 < tmp7;
                                auto tmp11 = tmp8 & tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr0[static_cast<long>(x3 + (240L*x2) + (3600L*x1) + (54000L*x0))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp16 = tmp0 >= tmp3;
                            auto tmp17 = static_cast<long>(480);
                            auto tmp18 = tmp0 < tmp17;
                            auto tmp19 = tmp16 & tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(1L + x1);
                                auto tmp22 = static_cast<long>(0);
                                auto tmp23 = tmp21 >= tmp22;
                                auto tmp24 = static_cast<long>(17);
                                auto tmp25 = tmp21 < tmp24;
                                auto tmp26 = c10::convert<long>(1L + x2);
                                auto tmp27 = tmp26 >= tmp22;
                                auto tmp28 = tmp26 < tmp24;
                                auto tmp29 = tmp23 & tmp25;
                                auto tmp30 = tmp29 & tmp27;
                                auto tmp31 = tmp30 & tmp28;
                                auto tmp32 = [&]
                                {
                                    auto tmp33 = in_ptr1[static_cast<long>(4080L + x3 + (240L*x2) + (4080L*x1) + (69360L*x0))];
                                    return tmp33;
                                }
                                ;
                                auto tmp34 = tmp31 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp36 = tmp0 >= tmp17;
                            auto tmp37 = static_cast<long>(720);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = tmp36 & tmp38;
                            auto tmp40 = [&]
                            {
                                auto tmp41 = c10::convert<long>(2L + x1);
                                auto tmp42 = static_cast<long>(0);
                                auto tmp43 = tmp41 >= tmp42;
                                auto tmp44 = static_cast<long>(19);
                                auto tmp45 = tmp41 < tmp44;
                                auto tmp46 = c10::convert<long>(2L + x2);
                                auto tmp47 = tmp46 >= tmp42;
                                auto tmp48 = tmp46 < tmp44;
                                auto tmp49 = tmp43 & tmp45;
                                auto tmp50 = tmp49 & tmp47;
                                auto tmp51 = tmp50 & tmp48;
                                auto tmp52 = [&]
                                {
                                    auto tmp53 = in_ptr2[static_cast<long>(9120L + x3 + (240L*x2) + (4560L*x1) + (86640L*x0))];
                                    return tmp53;
                                }
                                ;
                                auto tmp54 = tmp51 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                                return tmp54;
                            }
                            ;
                            auto tmp55 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                            auto tmp56 = tmp0 >= tmp37;
                            auto tmp57 = static_cast<long>(960);
                            auto tmp58 = tmp0 < tmp57;
                            auto tmp59 = [&]
                            {
                                auto tmp60 = c10::convert<long>(3L + x1);
                                auto tmp61 = static_cast<long>(0);
                                auto tmp62 = tmp60 >= tmp61;
                                auto tmp63 = static_cast<long>(21);
                                auto tmp64 = tmp60 < tmp63;
                                auto tmp65 = c10::convert<long>(3L + x2);
                                auto tmp66 = tmp65 >= tmp61;
                                auto tmp67 = tmp65 < tmp63;
                                auto tmp68 = tmp62 & tmp64;
                                auto tmp69 = tmp68 & tmp66;
                                auto tmp70 = tmp69 & tmp67;
                                auto tmp71 = [&]
                                {
                                    auto tmp72 = in_ptr3[static_cast<long>(15120L + x3 + (240L*x2) + (5040L*x1) + (105840L*x0))];
                                    return tmp72;
                                }
                                ;
                                auto tmp73 = tmp70 ? tmp71() : static_cast<decltype(tmp71())>(0.0);
                                return tmp73;
                            }
                            ;
                            auto tmp74 = tmp56 ? tmp59() : static_cast<decltype(tmp59())>(0.0);
                            auto tmp75 = tmp39 ? tmp55 : tmp74;
                            auto tmp76 = tmp19 ? tmp35 : tmp75;
                            auto tmp77 = tmp4 ? tmp15 : tmp76;
                            out_ptr0[static_cast<long>(x3 + (960L*x2) + (13440L*x1) + (188160L*x0))] = tmp77;
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp18.store(out_ptr2 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (480L*x2) + (94080L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(240);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (47040L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(480);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-47040L) + x2 + (196L*x1) + (47040L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_27 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (480L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (480L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (480L*x2) + (94080L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(240);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (47040L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(480);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-47040L) + x2 + (196L*x1) + (47040L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(196.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (94080L*x0))] = tmp29;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (480L*x2_inner) + (94080L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (94080L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_mul_native_batch_norm_backward_28 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr4[static_cast<long>(x1 + (480L*x2) + (94080L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(120);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (23520L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(240);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-23520L) + x2 + (196L*x1) + (23520L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(360);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = tmp15 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr2[static_cast<long>((-47040L) + x2 + (196L*x1) + (23520L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp0 >= tmp16;
                        auto tmp23 = static_cast<long>(480);
                        auto tmp24 = tmp0 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-70560L) + x2 + (196L*x1) + (23520L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        auto tmp29 = tmp11 ? tmp14 : tmp28;
                        auto tmp30 = tmp4 ? tmp7 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (94080L*x0))] = tmp32;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (480L*x2_inner) + (94080L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (94080L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_batch_norm_backward_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (160L*x1))];
                        auto tmp25 = in_ptr3[static_cast<long>(x0 + (160L*x1))];
                        auto tmp26 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(80);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x0 + (80L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(160);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-80L) + x0 + (80L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                        auto tmp17 = [&]
                        {
                            auto tmp18 = in_ptr1[static_cast<long>(x0 + (80L*x1))];
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr2[static_cast<long>((-80L) + x0 + (80L*x1))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp5 ? tmp19 : tmp22;
                        auto tmp24 = decltype(tmp0)(tmp0 + tmp23);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp28;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (160L*x0))];
                    auto tmp17 = in_ptr3[static_cast<long>(x1 + (160L*x0))];
                    auto tmp18 = in_ptr4[static_cast<long>(x1)];
                    auto tmp20 = out_ptr1[static_cast<long>(x1)];
                    auto tmp23 = in_ptr5[static_cast<long>(x1)];
                    auto tmp28 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(80);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (80L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(160);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-80L) + x1 + (80L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                    auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                    auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                    auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                    auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                    auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                    out_ptr2[static_cast<long>(x1 + (160L*x0))] = tmp33;
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (480L*x2) + (94080L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(240);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (47040L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(480);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-47040L) + x2 + (196L*x1) + (47040L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_32 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (480L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (480L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (480L*x2) + (94080L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(240);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (47040L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(480);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-47040L) + x2 + (196L*x1) + (47040L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(196.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (94080L*x0))] = tmp29;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (480L*x2_inner) + (94080L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (94080L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_mul_native_batch_norm_backward_33 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr4[static_cast<long>(x1 + (480L*x2) + (94080L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(120);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (23520L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(240);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-23520L) + x2 + (196L*x1) + (23520L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(360);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = tmp15 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr2[static_cast<long>((-47040L) + x2 + (196L*x1) + (23520L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp0 >= tmp16;
                        auto tmp23 = static_cast<long>(480);
                        auto tmp24 = tmp0 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-70560L) + x2 + (196L*x1) + (23520L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        auto tmp29 = tmp11 ? tmp14 : tmp28;
                        auto tmp30 = tmp4 ? tmp7 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (94080L*x0))] = tmp32;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (480L*x2_inner) + (94080L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (94080L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_convolution_backward_native_batch_norm_backward_34 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(80);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(160);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-80L) + x1 + (80L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp17 = [&]
                    {
                        auto tmp18 = in_ptr2[static_cast<long>(x1 + (80L*x0))];
                        return tmp18;
                    }
                    ;
                    auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-80L) + x1 + (80L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp5 ? tmp19 : tmp22;
                    auto tmp24 = decltype(tmp16)(tmp16 + tmp23);
                    in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp24;
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (160L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(160L); x2+=static_cast<long>(8L))
                    {
                        float tmp19[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (160L*x1) + (160L*x1_inner) + (31360L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (160L*x1) + (160L*x1_inner) + (31360L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                            tmp18.store(tmp19 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp19, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (31360L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(160L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (160L*x1) + (31360L*x0))];
                        auto tmp1 = in_ptr4[static_cast<long>(x2 + (160L*x1) + (31360L*x0))];
                        auto tmp2 = in_ptr5[static_cast<long>(x2)];
                        auto tmp4 = out_ptr1[static_cast<long>(x2)];
                        auto tmp7 = in_ptr6[static_cast<long>(x2)];
                        auto tmp12 = out_ptr0[static_cast<long>(x2)];
                        auto tmp15 = in_ptr7[static_cast<long>(x2)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp5 = static_cast<float>(0.0006377551020408163);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp7)(tmp7 * tmp7);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                        auto tmp11 = decltype(tmp0)(tmp0 - tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp5);
                        auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                        auto tmp16 = decltype(tmp7)(tmp7 * tmp15);
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (31360L*x0))] = tmp17;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(15680L + x2 + (196L*x1) + (196L*x1_inner) + (31360L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)), static_cast<long>(80L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(15680L + x2 + (196L*x1) + (196L*x1_inner) + (31360L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (31360L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)), static_cast<long>(80L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (31360L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (480L*x2) + (94080L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(240);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (47040L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(480);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-47040L) + x2 + (196L*x1) + (47040L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_38 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (480L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (480L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (480L*x2) + (94080L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(240);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (47040L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(480);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-47040L) + x2 + (196L*x1) + (47040L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(196.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (94080L*x0))] = tmp29;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (480L*x2_inner) + (94080L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (94080L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_mul_native_batch_norm_backward_39 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr4[static_cast<long>(x1 + (480L*x2) + (94080L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(120);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (23520L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(240);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-23520L) + x2 + (196L*x1) + (23520L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(360);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = tmp15 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr2[static_cast<long>((-47040L) + x2 + (196L*x1) + (23520L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp0 >= tmp16;
                        auto tmp23 = static_cast<long>(480);
                        auto tmp24 = tmp0 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-70560L) + x2 + (196L*x1) + (23520L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        auto tmp29 = tmp11 ? tmp14 : tmp28;
                        auto tmp30 = tmp4 ? tmp7 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (94080L*x0))] = tmp32;
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (94080L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (480L*x2_inner) + (94080L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (94080L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_batch_norm_backward_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (160L*x1))];
                        auto tmp25 = in_ptr3[static_cast<long>(x0 + (160L*x1))];
                        auto tmp26 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(80);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x0 + (80L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(160);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-80L) + x0 + (80L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                        auto tmp17 = [&]
                        {
                            auto tmp18 = in_ptr1[static_cast<long>(x0 + (80L*x1))];
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr2[static_cast<long>((-80L) + x0 + (80L*x1))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp5 ? tmp19 : tmp22;
                        auto tmp24 = decltype(tmp0)(tmp0 + tmp23);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp28;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                    auto tmp17 = in_ptr3[static_cast<long>(x1 + (160L*x0))];
                    auto tmp18 = in_ptr4[static_cast<long>(x1)];
                    auto tmp20 = out_ptr1[static_cast<long>(x1)];
                    auto tmp23 = in_ptr5[static_cast<long>(x1)];
                    auto tmp28 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(80);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (80L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(160);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-80L) + x1 + (80L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                    auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                    auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                    auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                    auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                    auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                    in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp33;
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (624L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4992L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(416L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_43 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (624L*x2) + (122304L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (624L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (624L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (624L*x2) + (122304L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (624L*x2) + (122304L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(196.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp10 = decltype(tmp9)(1)/(decltype(tmp9)(1) + tmp9.neg().exp());
                            auto tmp11 = static_cast<float>(1.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp12 - tmp10;
                            auto tmp14 = tmp9 * tmp13;
                            auto tmp15 = tmp14 + tmp12;
                            auto tmp16 = tmp10 * tmp15;
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp20 = tmp18 - tmp19;
                            auto tmp21 = tmp17 * tmp20;
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                            tmp_acc1_vec = tmp_acc1_vec + tmp21;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(624L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (624L*x1) + (122304L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (624L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (624L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (624L*x1) + (122304L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (624L*x1) + (122304L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(196.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp3 + tmp7;
                        auto tmp10 = decltype(tmp9)(1)/(decltype(tmp9)(1) + tmp9.neg().exp());
                        auto tmp11 = static_cast<float>(1.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp12 - tmp10;
                        auto tmp14 = tmp9 * tmp13;
                        auto tmp15 = tmp14 + tmp12;
                        auto tmp16 = tmp10 * tmp15;
                        auto tmp17 = tmp8 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp22 = static_cast<float>(0.0006377551020408163);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (624L*x1) + (122304L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (624L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (624L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (624L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (624L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (624L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (624L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (624L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (624L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (624L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (104L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (104L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (104L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (104L*x0)));
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
                    tmp18.store(out_ptr2 + static_cast<long>(x1 + (104L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (624L*x2) + (122304L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(312);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (61152L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(624);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-61152L) + x2 + (196L*x1) + (61152L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (624L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4992L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(208L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_48 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (624L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (624L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (624L*x2) + (122304L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(312);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (61152L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(624);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-61152L) + x2 + (196L*x1) + (61152L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(196.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (122304L*x0))] = tmp29;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (624L*x2_inner) + (122304L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (122304L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (122304L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_mul_native_batch_norm_backward_49 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr4[static_cast<long>(x1 + (624L*x2) + (122304L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(156);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (30576L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(312);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-30576L) + x2 + (196L*x1) + (30576L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(468);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = tmp15 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr2[static_cast<long>((-61152L) + x2 + (196L*x1) + (30576L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp0 >= tmp16;
                        auto tmp23 = static_cast<long>(624);
                        auto tmp24 = tmp0 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-91728L) + x2 + (196L*x1) + (30576L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        auto tmp29 = tmp11 ? tmp14 : tmp28;
                        auto tmp30 = tmp4 ? tmp7 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (122304L*x0))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (624L*x2_inner) + (122304L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (122304L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (122304L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_batch_norm_backward_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (104L*x1))];
                        auto tmp25 = in_ptr3[static_cast<long>(x0 + (104L*x1))];
                        auto tmp26 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(52);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x0 + (52L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(104);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-52L) + x0 + (52L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                        auto tmp17 = [&]
                        {
                            auto tmp18 = in_ptr1[static_cast<long>(x0 + (52L*x1))];
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr2[static_cast<long>((-52L) + x0 + (52L*x1))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp5 ? tmp19 : tmp22;
                        auto tmp24 = decltype(tmp0)(tmp0 + tmp23);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp28;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(104L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (104L*x0))];
                    auto tmp17 = in_ptr3[static_cast<long>(x1 + (104L*x0))];
                    auto tmp18 = in_ptr4[static_cast<long>(x1)];
                    auto tmp20 = out_ptr1[static_cast<long>(x1)];
                    auto tmp23 = in_ptr5[static_cast<long>(x1)];
                    auto tmp28 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(52);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (52L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(104);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-52L) + x1 + (52L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                    auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                    auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                    auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                    auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                    auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                    out_ptr2[static_cast<long>(x1 + (104L*x0))] = tmp33;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (624L*x2) + (122304L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(312);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (61152L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(624);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-61152L) + x2 + (196L*x1) + (61152L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (624L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4992L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(208L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_53 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (624L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (624L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (624L*x2) + (122304L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(312);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (61152L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(624);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-61152L) + x2 + (196L*x1) + (61152L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(196.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (122304L*x0))] = tmp29;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (624L*x2_inner) + (122304L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (122304L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (122304L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_mul_native_batch_norm_backward_54 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr4[static_cast<long>(x1 + (624L*x2) + (122304L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(156);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (30576L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(312);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-30576L) + x2 + (196L*x1) + (30576L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(468);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = tmp15 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr2[static_cast<long>((-61152L) + x2 + (196L*x1) + (30576L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp0 >= tmp16;
                        auto tmp23 = static_cast<long>(624);
                        auto tmp24 = tmp0 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-91728L) + x2 + (196L*x1) + (30576L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        auto tmp29 = tmp11 ? tmp14 : tmp28;
                        auto tmp30 = tmp4 ? tmp7 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (122304L*x0))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (624L*x2_inner) + (122304L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (122304L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (122304L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_convolution_backward_native_batch_norm_backward_55 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(104L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (104L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(52);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr0[static_cast<long>(x1 + (52L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(104);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-52L) + x1 + (52L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp17 = [&]
                    {
                        auto tmp18 = in_ptr2[static_cast<long>(x1 + (52L*x0))];
                        return tmp18;
                    }
                    ;
                    auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-52L) + x1 + (52L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp5 ? tmp19 : tmp22;
                    auto tmp24 = decltype(tmp16)(tmp16 + tmp23);
                    in_out_ptr0[static_cast<long>(x1 + (104L*x0))] = tmp24;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (104L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (104L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(104L); x2+=static_cast<long>(8L))
                    {
                        float tmp19[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (104L*x1) + (104L*x1_inner) + (20384L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (104L*x1) + (104L*x1_inner) + (20384L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                            tmp18.store(tmp19 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp19, 8, out_ptr2 + static_cast<long>(x1 + (196L*x2) + (20384L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(104L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (104L*x1) + (20384L*x0))];
                        auto tmp1 = in_ptr4[static_cast<long>(x2 + (104L*x1) + (20384L*x0))];
                        auto tmp2 = in_ptr5[static_cast<long>(x2)];
                        auto tmp4 = out_ptr1[static_cast<long>(x2)];
                        auto tmp7 = in_ptr6[static_cast<long>(x2)];
                        auto tmp12 = out_ptr0[static_cast<long>(x2)];
                        auto tmp15 = in_ptr7[static_cast<long>(x2)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp5 = static_cast<float>(0.0006377551020408163);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp7)(tmp7 * tmp7);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                        auto tmp11 = decltype(tmp0)(tmp0 - tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp5);
                        auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                        auto tmp16 = decltype(tmp7)(tmp7 * tmp15);
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (20384L*x0))] = tmp17;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(10192L + x2 + (196L*x1) + (196L*x1_inner) + (20384L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (52L*x2) + (10192L*x0)), static_cast<long>(52L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(10192L + x2 + (196L*x1) + (196L*x1_inner) + (20384L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr3 + static_cast<long>(x1 + (52L*x2) + (10192L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(52L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(10192L + x2 + (196L*x1) + (20384L*x0))];
                            out_ptr3[static_cast<long>(x1 + (52L*x2) + (10192L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (20384L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (52L*x2) + (10192L*x0)), static_cast<long>(52L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (20384L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (52L*x2) + (10192L*x0)));
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(52L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (20384L*x0))];
                    out_ptr0[static_cast<long>(x1 + (52L*x2) + (10192L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (624L*x2) + (122304L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(312);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (61152L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(624);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-61152L) + x2 + (196L*x1) + (61152L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (624L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4992L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(208L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_59 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (624L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (624L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (624L*x2) + (122304L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(312);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (61152L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(624);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-61152L) + x2 + (196L*x1) + (61152L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(196.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (122304L*x0))] = tmp29;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (624L*x2_inner) + (122304L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (122304L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (122304L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_mul_native_batch_norm_backward_60 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr4[static_cast<long>(x1 + (624L*x2) + (122304L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(156);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (30576L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(312);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-30576L) + x2 + (196L*x1) + (30576L*x0))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(468);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = tmp15 & tmp17;
                        auto tmp19 = [&]
                        {
                            auto tmp20 = in_ptr2[static_cast<long>((-61152L) + x2 + (196L*x1) + (30576L*x0))];
                            return tmp20;
                        }
                        ;
                        auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                        auto tmp22 = tmp0 >= tmp16;
                        auto tmp23 = static_cast<long>(624);
                        auto tmp24 = tmp0 < tmp23;
                        auto tmp25 = [&]
                        {
                            auto tmp26 = in_ptr3[static_cast<long>((-91728L) + x2 + (196L*x1) + (30576L*x0))];
                            return tmp26;
                        }
                        ;
                        auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                        auto tmp28 = tmp18 ? tmp21 : tmp27;
                        auto tmp29 = tmp11 ? tmp14 : tmp28;
                        auto tmp30 = tmp4 ? tmp7 : tmp29;
                        auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (122304L*x0))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (196L*x0) + (122304L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (624L*x2_inner) + (122304L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (122304L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (624L*x2) + (122304L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.0006377551020408163);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(624L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_batch_norm_backward_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (104L*x1))];
                        auto tmp25 = in_ptr3[static_cast<long>(x0 + (104L*x1))];
                        auto tmp26 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(52);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x0 + (52L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(104);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-52L) + x0 + (52L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                        auto tmp17 = [&]
                        {
                            auto tmp18 = in_ptr1[static_cast<long>(x0 + (52L*x1))];
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr2[static_cast<long>((-52L) + x0 + (52L*x1))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp5 ? tmp19 : tmp22;
                        auto tmp24 = decltype(tmp0)(tmp0 + tmp23);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp28;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(104L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (104L*x0))];
                    auto tmp17 = in_ptr3[static_cast<long>(x1 + (104L*x0))];
                    auto tmp18 = in_ptr4[static_cast<long>(x1)];
                    auto tmp20 = out_ptr1[static_cast<long>(x1)];
                    auto tmp23 = in_ptr5[static_cast<long>(x1)];
                    auto tmp28 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(52);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (52L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(104);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-52L) + x1 + (52L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                    auto tmp21 = static_cast<float>(0.0006377551020408163);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                    auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                    auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                    auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                    auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                    auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                    in_out_ptr0[static_cast<long>(x1 + (104L*x0))] = tmp33;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (336L*x2) + (65856L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (336L*x2) + (65856L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (336L*x2) + (65856L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (336L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (336L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (336L*x2) + (65856L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (336L*x2) + (65856L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(196.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp10 = decltype(tmp9)(1)/(decltype(tmp9)(1) + tmp9.neg().exp());
                            auto tmp11 = static_cast<float>(1.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp12 - tmp10;
                            auto tmp14 = tmp9 * tmp13;
                            auto tmp15 = tmp14 + tmp12;
                            auto tmp16 = tmp10 * tmp15;
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp20 = tmp18 - tmp19;
                            auto tmp21 = tmp17 * tmp20;
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                            tmp_acc1_vec = tmp_acc1_vec + tmp21;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(336L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (336L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (336L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(196.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp3 + tmp7;
                        auto tmp10 = decltype(tmp9)(1)/(decltype(tmp9)(1) + tmp9.neg().exp());
                        auto tmp11 = static_cast<float>(1.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp12 - tmp10;
                        auto tmp14 = tmp9 * tmp13;
                        auto tmp15 = tmp14 + tmp12;
                        auto tmp16 = tmp10 * tmp15;
                        auto tmp17 = tmp8 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp22 = static_cast<float>(0.0006377551020408163);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(224L + x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(224L + x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(224L + x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr2 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(112L + x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(112L + x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(112L + x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_67 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(336L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(112);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x1);
                                auto tmp7 = static_cast<long>(29);
                                auto tmp8 = tmp6 < tmp7;
                                auto tmp9 = c10::convert<long>(x2);
                                auto tmp10 = tmp9 < tmp7;
                                auto tmp11 = tmp8 & tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr0[static_cast<long>(x3 + (112L*x2) + (3248L*x1) + (94192L*x0))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp16 = tmp0 >= tmp3;
                            auto tmp17 = static_cast<long>(224);
                            auto tmp18 = tmp0 < tmp17;
                            auto tmp19 = tmp16 & tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(1L + x1);
                                auto tmp22 = static_cast<long>(0);
                                auto tmp23 = tmp21 >= tmp22;
                                auto tmp24 = static_cast<long>(31);
                                auto tmp25 = tmp21 < tmp24;
                                auto tmp26 = c10::convert<long>(1L + x2);
                                auto tmp27 = tmp26 >= tmp22;
                                auto tmp28 = tmp26 < tmp24;
                                auto tmp29 = tmp23 & tmp25;
                                auto tmp30 = tmp29 & tmp27;
                                auto tmp31 = tmp30 & tmp28;
                                auto tmp32 = [&]
                                {
                                    auto tmp33 = in_ptr1[static_cast<long>(3472L + x3 + (112L*x2) + (3472L*x1) + (107632L*x0))];
                                    return tmp33;
                                }
                                ;
                                auto tmp34 = tmp31 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp36 = tmp0 >= tmp17;
                            auto tmp37 = static_cast<long>(336);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(2L + x1);
                                auto tmp41 = static_cast<long>(0);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = static_cast<long>(33);
                                auto tmp44 = tmp40 < tmp43;
                                auto tmp45 = c10::convert<long>(2L + x2);
                                auto tmp46 = tmp45 >= tmp41;
                                auto tmp47 = tmp45 < tmp43;
                                auto tmp48 = tmp42 & tmp44;
                                auto tmp49 = tmp48 & tmp46;
                                auto tmp50 = tmp49 & tmp47;
                                auto tmp51 = [&]
                                {
                                    auto tmp52 = in_ptr2[static_cast<long>(7392L + x3 + (112L*x2) + (3696L*x1) + (121968L*x0))];
                                    return tmp52;
                                }
                                ;
                                auto tmp53 = tmp50 ? tmp51() : static_cast<decltype(tmp51())>(0.0);
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp55 = tmp19 ? tmp35 : tmp54;
                            auto tmp56 = tmp4 ? tmp15 : tmp55;
                            out_ptr0[static_cast<long>(x3 + (336L*x2) + (9408L*x1) + (263424L*x0))] = tmp56;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (336L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (336L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (336L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (56L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (56L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (56L*x0)));
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
                    tmp18.store(out_ptr2 + static_cast<long>(x1 + (56L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(168);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (131712L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(336);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x1) + (131712L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (336L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (336L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(168);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (131712L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(336);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x1) + (131712L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(784.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (784L*x1) + (263424L*x0))] = tmp29;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (784L*x0) + (263424L*x1)), static_cast<long>(784L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (784L*x0) + (263424L*x1)), static_cast<long>(784L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (336L*x2) + (336L*x2_inner) + (263424L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.00015943877551020407);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_mul_native_batch_norm_backward_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x0 + (336L*x2) + (263424L*x1))];
                            auto tmp25 = in_ptr3[static_cast<long>(x0 + (336L*x2) + (263424L*x1))];
                            auto tmp26 = in_ptr4[static_cast<long>(x0)];
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(168);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x0) + (131712L*x1))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(336);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x0) + (131712L*x1))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>(x2 + (784L*x0) + (131712L*x1))];
                                return tmp18;
                            }
                            ;
                            auto tmp19 = tmp4 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x0) + (131712L*x1))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp8 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp23 = tmp4 ? tmp19 : tmp22;
                            auto tmp24 = decltype(tmp23)(tmp23 * tmp15);
                            auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                            auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                            tmp_acc0 = tmp_acc0 + tmp16;
                            tmp_acc1 = tmp_acc1 + tmp28;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                        auto tmp18 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = out_ptr1[static_cast<long>(x1)];
                        auto tmp23 = in_ptr5[static_cast<long>(x1)];
                        auto tmp28 = out_ptr0[static_cast<long>(x1)];
                        auto tmp31 = in_ptr6[static_cast<long>(x1)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(168);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (131712L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(336);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x1) + (131712L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                        auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                        auto tmp21 = static_cast<float>(0.00015943877551020407);
                        auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                        auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                        auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                        auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                        auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                        auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                        auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                        auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                        auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                        out_ptr2[static_cast<long>(x2 + (784L*x1) + (263424L*x0))] = tmp33;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_batch_norm_backward_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (56L*x1))];
                        auto tmp25 = in_ptr3[static_cast<long>(x0 + (56L*x1))];
                        auto tmp26 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(28);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x0 + (28L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(56);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-28L) + x0 + (28L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                        auto tmp17 = [&]
                        {
                            auto tmp18 = in_ptr1[static_cast<long>(x0 + (28L*x1))];
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr2[static_cast<long>((-28L) + x0 + (28L*x1))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp5 ? tmp19 : tmp22;
                        auto tmp24 = decltype(tmp0)(tmp0 + tmp23);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp28;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (56L*x0))];
                    auto tmp17 = in_ptr3[static_cast<long>(x1 + (56L*x0))];
                    auto tmp18 = in_ptr4[static_cast<long>(x1)];
                    auto tmp20 = out_ptr1[static_cast<long>(x1)];
                    auto tmp23 = in_ptr5[static_cast<long>(x1)];
                    auto tmp28 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(28);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (28L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(56);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-28L) + x1 + (28L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                    auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                    auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                    auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                    auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                    auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                    out_ptr2[static_cast<long>(x1 + (56L*x0))] = tmp33;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(168);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (131712L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(336);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x1) + (131712L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_76 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (336L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (336L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(168);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (131712L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(336);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x1) + (131712L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(784.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (784L*x1) + (263424L*x0))] = tmp29;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (784L*x0) + (263424L*x1)), static_cast<long>(784L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (784L*x0) + (263424L*x1)), static_cast<long>(784L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (336L*x2) + (336L*x2_inner) + (263424L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.00015943877551020407);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_mul_native_batch_norm_backward_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x0 + (336L*x2) + (263424L*x1))];
                            auto tmp25 = in_ptr3[static_cast<long>(x0 + (336L*x2) + (263424L*x1))];
                            auto tmp26 = in_ptr4[static_cast<long>(x0)];
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(168);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x0) + (131712L*x1))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(336);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x0) + (131712L*x1))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>(x2 + (784L*x0) + (131712L*x1))];
                                return tmp18;
                            }
                            ;
                            auto tmp19 = tmp4 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x0) + (131712L*x1))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp8 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp23 = tmp4 ? tmp19 : tmp22;
                            auto tmp24 = decltype(tmp23)(tmp23 * tmp15);
                            auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                            auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                            tmp_acc0 = tmp_acc0 + tmp16;
                            tmp_acc1 = tmp_acc1 + tmp28;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                        auto tmp18 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = out_ptr1[static_cast<long>(x1)];
                        auto tmp23 = in_ptr5[static_cast<long>(x1)];
                        auto tmp28 = out_ptr0[static_cast<long>(x1)];
                        auto tmp31 = in_ptr6[static_cast<long>(x1)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(168);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (131712L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(336);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x1) + (131712L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                        auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                        auto tmp21 = static_cast<float>(0.00015943877551020407);
                        auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                        auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                        auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                        auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                        auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                        auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                        auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                        auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                        auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                        out_ptr2[static_cast<long>(x2 + (784L*x1) + (263424L*x0))] = tmp33;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_convolution_backward_native_batch_norm_backward_78 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (56L*x0))];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(28);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr0[static_cast<long>(x1 + (28L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(56);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-28L) + x1 + (28L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp17 = [&]
                    {
                        auto tmp18 = in_ptr2[static_cast<long>(x1 + (28L*x0))];
                        return tmp18;
                    }
                    ;
                    auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((-28L) + x1 + (28L*x0))];
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp23 = tmp5 ? tmp19 : tmp22;
                    auto tmp24 = decltype(tmp16)(tmp16 + tmp23);
                    in_out_ptr0[static_cast<long>(x1 + (56L*x0))] = tmp24;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (56L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (56L*x1)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(8L))
                    {
                        float tmp19[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (56L*x1) + (56L*x1_inner) + (43904L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (56L*x1) + (56L*x1_inner) + (43904L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                            tmp18.store(tmp19 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp19, 8, out_ptr2 + static_cast<long>(x1 + (784L*x2) + (43904L*x0)), static_cast<long>(784L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(21952L + x2 + (784L*x1) + (784L*x1_inner) + (43904L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (28L*x2) + (21952L*x0)), static_cast<long>(28L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(21952L + x2 + (784L*x1) + (43904L*x0))];
                        out_ptr3[static_cast<long>(x1 + (28L*x2) + (21952L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (43904L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (28L*x2) + (21952L*x0)), static_cast<long>(28L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (43904L*x0))];
                        out_ptr0[static_cast<long>(x1 + (28L*x2) + (21952L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(168);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (131712L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(336);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x1) + (131712L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp18 = decltype(tmp14)(tmp14 * tmp17);
                            tmp_acc0 = tmp_acc0 + tmp18;
                        }
                        out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_82 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (336L*x0))];
                        auto tmp18 = in_ptr3[static_cast<long>(x1 + (336L*x0))];
                        auto tmp22 = in_ptr4[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(168);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (131712L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(336);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x1) + (131712L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp15)(1) / (decltype(tmp15)(1) + std::exp(-tmp15));
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        auto tmp19 = static_cast<float>(784.0);
                        auto tmp20 = tmp18 / tmp19;
                        auto tmp21 = decltype(tmp17)(tmp17 + tmp20);
                        auto tmp23 = decltype(tmp22)(1) / (decltype(tmp22)(1) + std::exp(-tmp22));
                        auto tmp24 = static_cast<float>(1.0);
                        auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp24);
                        auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                        auto tmp29 = decltype(tmp21)(tmp21 * tmp28);
                        out_ptr0[static_cast<long>(x2 + (784L*x1) + (263424L*x0))] = tmp29;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (784L*x0) + (263424L*x1)), static_cast<long>(784L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x2 + (784L*x0) + (263424L*x1)), static_cast<long>(784L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (336L*x2) + (336L*x2_inner) + (263424L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(0.00015943877551020407);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                            auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp14 = tmp0 - tmp13;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp7);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp9)(tmp9 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_mul_native_batch_norm_backward_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp15 = in_ptr2[static_cast<long>(x0 + (336L*x2) + (263424L*x1))];
                            auto tmp25 = in_ptr3[static_cast<long>(x0 + (336L*x2) + (263424L*x1))];
                            auto tmp26 = in_ptr4[static_cast<long>(x0)];
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(168);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x0) + (131712L*x1))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(336);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x0) + (131712L*x1))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>(x2 + (784L*x0) + (131712L*x1))];
                                return tmp18;
                            }
                            ;
                            auto tmp19 = tmp4 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x0) + (131712L*x1))];
                                return tmp21;
                            }
                            ;
                            auto tmp22 = tmp8 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp23 = tmp4 ? tmp19 : tmp22;
                            auto tmp24 = decltype(tmp23)(tmp23 * tmp15);
                            auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                            auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                            tmp_acc0 = tmp_acc0 + tmp16;
                            tmp_acc1 = tmp_acc1 + tmp28;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                        auto tmp17 = in_ptr3[static_cast<long>(x1 + (336L*x2) + (263424L*x0))];
                        auto tmp18 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = out_ptr1[static_cast<long>(x1)];
                        auto tmp23 = in_ptr5[static_cast<long>(x1)];
                        auto tmp28 = out_ptr0[static_cast<long>(x1)];
                        auto tmp31 = in_ptr6[static_cast<long>(x1)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(168);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (131712L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(336);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-131712L) + x2 + (784L*x1) + (131712L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                        auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                        auto tmp21 = static_cast<float>(0.00015943877551020407);
                        auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                        auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                        auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                        auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                        auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                        auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                        auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                        auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                        auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                        out_ptr2[static_cast<long>(x2 + (784L*x1) + (263424L*x0))] = tmp33;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_batch_norm_backward_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (56L*x1))];
                        auto tmp25 = in_ptr3[static_cast<long>(x0 + (56L*x1))];
                        auto tmp26 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(28);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x0 + (28L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(56);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-28L) + x0 + (28L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                        auto tmp17 = [&]
                        {
                            auto tmp18 = in_ptr1[static_cast<long>(x0 + (28L*x1))];
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr2[static_cast<long>((-28L) + x0 + (28L*x1))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp5 ? tmp19 : tmp22;
                        auto tmp24 = decltype(tmp0)(tmp0 + tmp23);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp28;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (56L*x0))];
                    auto tmp17 = in_ptr3[static_cast<long>(x1 + (56L*x0))];
                    auto tmp18 = in_ptr4[static_cast<long>(x1)];
                    auto tmp20 = out_ptr1[static_cast<long>(x1)];
                    auto tmp23 = in_ptr5[static_cast<long>(x1)];
                    auto tmp28 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(28);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (28L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(56);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-28L) + x1 + (28L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                    auto tmp21 = static_cast<float>(0.00015943877551020407);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                    auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                    auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                    auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                    auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                    auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                    in_out_ptr0[static_cast<long>(x1 + (56L*x0))] = tmp33;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x2) + (188160L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x2) + (188160L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp4 - tmp2;
            auto tmp6 = tmp1 * tmp5;
            auto tmp7 = tmp6 + tmp4;
            auto tmp8 = tmp2 * tmp7;
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_87 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x2) + (188160L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x2) + (188160L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (240L*x2) + (188160L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(784.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp10 = decltype(tmp9)(1)/(decltype(tmp9)(1) + tmp9.neg().exp());
                            auto tmp11 = static_cast<float>(1.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp12 - tmp10;
                            auto tmp14 = tmp9 * tmp13;
                            auto tmp15 = tmp14 + tmp12;
                            auto tmp16 = tmp10 * tmp15;
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp20 = tmp18 - tmp19;
                            auto tmp21 = tmp17 * tmp20;
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                            tmp_acc1_vec = tmp_acc1_vec + tmp21;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (188160L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (188160L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (240L*x1) + (188160L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(784.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp3 + tmp7;
                        auto tmp10 = decltype(tmp9)(1)/(decltype(tmp9)(1) + tmp9.neg().exp());
                        auto tmp11 = static_cast<float>(1.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp12 - tmp10;
                        auto tmp14 = tmp9 * tmp13;
                        auto tmp15 = tmp14 + tmp12;
                        auto tmp16 = tmp10 * tmp15;
                        auto tmp17 = tmp8 * tmp16;
                        auto tmp20 = tmp18 - tmp19;
                        auto tmp22 = static_cast<float>(0.00015943877551020407);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (188160L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(180L + x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(180L + x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(180L + x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr2 + static_cast<long>(x1 + (60L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(180L + x1 + (240L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(180L + x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(180L + x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    out_ptr2[static_cast<long>(x1 + (60L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(120L + x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(120L + x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(120L + x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (60L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(120L + x1 + (240L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(120L + x1)];
                    auto tmp2 = in_ptr2[static_cast<long>(120L + x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    out_ptr0[static_cast<long>(x1 + (60L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(60L + x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(60L + x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(60L + x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (60L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(60L + x1 + (240L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(60L + x1)];
                    auto tmp2 = in_ptr2[static_cast<long>(60L + x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    out_ptr0[static_cast<long>(x1 + (60L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (60L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (240L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1)];
                    auto tmp2 = in_ptr2[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    out_ptr0[static_cast<long>(x1 + (60L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_91 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(60);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x1);
                                auto tmp7 = static_cast<long>(57);
                                auto tmp8 = tmp6 < tmp7;
                                auto tmp9 = c10::convert<long>(x2);
                                auto tmp10 = tmp9 < tmp7;
                                auto tmp11 = tmp8 & tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr0[static_cast<long>(x3 + (60L*x2) + (3420L*x1) + (194940L*x0))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp16 = tmp0 >= tmp3;
                            auto tmp17 = static_cast<long>(120);
                            auto tmp18 = tmp0 < tmp17;
                            auto tmp19 = tmp16 & tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(1L + x1);
                                auto tmp22 = static_cast<long>(0);
                                auto tmp23 = tmp21 >= tmp22;
                                auto tmp24 = static_cast<long>(59);
                                auto tmp25 = tmp21 < tmp24;
                                auto tmp26 = c10::convert<long>(1L + x2);
                                auto tmp27 = tmp26 >= tmp22;
                                auto tmp28 = tmp26 < tmp24;
                                auto tmp29 = tmp23 & tmp25;
                                auto tmp30 = tmp29 & tmp27;
                                auto tmp31 = tmp30 & tmp28;
                                auto tmp32 = [&]
                                {
                                    auto tmp33 = in_ptr1[static_cast<long>(3540L + x3 + (60L*x2) + (3540L*x1) + (208860L*x0))];
                                    return tmp33;
                                }
                                ;
                                auto tmp34 = tmp31 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp36 = tmp0 >= tmp17;
                            auto tmp37 = static_cast<long>(180);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = tmp36 & tmp38;
                            auto tmp40 = [&]
                            {
                                auto tmp41 = c10::convert<long>(2L + x1);
                                auto tmp42 = static_cast<long>(0);
                                auto tmp43 = tmp41 >= tmp42;
                                auto tmp44 = static_cast<long>(61);
                                auto tmp45 = tmp41 < tmp44;
                                auto tmp46 = c10::convert<long>(2L + x2);
                                auto tmp47 = tmp46 >= tmp42;
                                auto tmp48 = tmp46 < tmp44;
                                auto tmp49 = tmp43 & tmp45;
                                auto tmp50 = tmp49 & tmp47;
                                auto tmp51 = tmp50 & tmp48;
                                auto tmp52 = [&]
                                {
                                    auto tmp53 = in_ptr2[static_cast<long>(7320L + x3 + (60L*x2) + (3660L*x1) + (223260L*x0))];
                                    return tmp53;
                                }
                                ;
                                auto tmp54 = tmp51 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                                return tmp54;
                            }
                            ;
                            auto tmp55 = tmp39 ? tmp40() : static_cast<decltype(tmp40())>(0.0);
                            auto tmp56 = tmp0 >= tmp37;
                            auto tmp57 = static_cast<long>(240);
                            auto tmp58 = tmp0 < tmp57;
                            auto tmp59 = [&]
                            {
                                auto tmp60 = c10::convert<long>(3L + x1);
                                auto tmp61 = static_cast<long>(0);
                                auto tmp62 = tmp60 >= tmp61;
                                auto tmp63 = static_cast<long>(63);
                                auto tmp64 = tmp60 < tmp63;
                                auto tmp65 = c10::convert<long>(3L + x2);
                                auto tmp66 = tmp65 >= tmp61;
                                auto tmp67 = tmp65 < tmp63;
                                auto tmp68 = tmp62 & tmp64;
                                auto tmp69 = tmp68 & tmp66;
                                auto tmp70 = tmp69 & tmp67;
                                auto tmp71 = [&]
                                {
                                    auto tmp72 = in_ptr3[static_cast<long>(11340L + x3 + (60L*x2) + (3780L*x1) + (238140L*x0))];
                                    return tmp72;
                                }
                                ;
                                auto tmp73 = tmp70 ? tmp71() : static_cast<decltype(tmp71())>(0.0);
                                return tmp73;
                            }
                            ;
                            auto tmp74 = tmp56 ? tmp59() : static_cast<decltype(tmp59())>(0.0);
                            auto tmp75 = tmp39 ? tmp55 : tmp74;
                            auto tmp76 = tmp19 ? tmp35 : tmp75;
                            auto tmp77 = tmp4 ? tmp15 : tmp76;
                            out_ptr0[static_cast<long>(x3 + (240L*x2) + (13440L*x1) + (752640L*x0))] = tmp77;
                        }
                    }
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
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
                    tmp18.store(out_ptr2 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_batch_norm_backward_threshold_backward_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (120L*x1))];
                        auto tmp26 = in_ptr3[static_cast<long>(x0 + (120L*x1))];
                        auto tmp27 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(60);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x0 + (60L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(120);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-60L) + x0 + (60L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = tmp0 ? tmp16 : tmp15;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr1[static_cast<long>(x0 + (60L*x1))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp5 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = [&]
                        {
                            auto tmp22 = in_ptr2[static_cast<long>((-60L) + x0 + (60L*x1))];
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp9 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp24 = tmp5 ? tmp20 : tmp23;
                        auto tmp25 = tmp0 ? tmp16 : tmp24;
                        auto tmp28 = decltype(tmp26)(tmp26 - tmp27);
                        auto tmp29 = decltype(tmp25)(tmp25 * tmp28);
                        tmp_acc0 = tmp_acc0 + tmp17;
                        tmp_acc1 = tmp_acc1 + tmp29;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (120L*x0))];
                    auto tmp18 = in_ptr3[static_cast<long>(x1 + (120L*x0))];
                    auto tmp19 = in_ptr4[static_cast<long>(x1)];
                    auto tmp21 = out_ptr1[static_cast<long>(x1)];
                    auto tmp24 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = out_ptr0[static_cast<long>(x1)];
                    auto tmp32 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(60);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (60L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(120);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-60L) + x1 + (60L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp0 ? tmp16 : tmp15;
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    auto tmp22 = static_cast<float>(3.985969387755102e-05);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp25 = decltype(tmp24)(tmp24 * tmp24);
                    auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                    auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                    auto tmp28 = decltype(tmp17)(tmp17 - tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp22);
                    auto tmp31 = decltype(tmp28)(tmp28 - tmp30);
                    auto tmp33 = decltype(tmp24)(tmp24 * tmp32);
                    auto tmp34 = decltype(tmp31)(tmp31 * tmp33);
                    out_ptr2[static_cast<long>(x1 + (120L*x0))] = tmp34;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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


cpp_fused_native_batch_norm_backward_threshold_backward_94 = async_compile.cpp('''
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
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_batch_norm_backward_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (40L*x1))];
                        auto tmp25 = in_ptr3[static_cast<long>(x0 + (40L*x1))];
                        auto tmp26 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(20);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x0 + (20L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(40);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-20L) + x0 + (20L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                        auto tmp17 = [&]
                        {
                            auto tmp18 = in_ptr1[static_cast<long>(x0 + (20L*x1))];
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp5 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr2[static_cast<long>((-20L) + x0 + (20L*x1))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp9 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp5 ? tmp19 : tmp22;
                        auto tmp24 = decltype(tmp0)(tmp0 + tmp23);
                        auto tmp27 = decltype(tmp25)(tmp25 - tmp26);
                        auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp28;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (40L*x0))];
                    auto tmp17 = in_ptr3[static_cast<long>(x1 + (40L*x0))];
                    auto tmp18 = in_ptr4[static_cast<long>(x1)];
                    auto tmp20 = out_ptr1[static_cast<long>(x1)];
                    auto tmp23 = in_ptr5[static_cast<long>(x1)];
                    auto tmp28 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(20);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (20L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(40);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-20L) + x1 + (20L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    auto tmp19 = decltype(tmp17)(tmp17 - tmp18);
                    auto tmp21 = static_cast<float>(3.985969387755102e-05);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp23);
                    auto tmp25 = decltype(tmp22)(tmp22 * tmp24);
                    auto tmp26 = decltype(tmp19)(tmp19 * tmp25);
                    auto tmp27 = decltype(tmp16)(tmp16 - tmp26);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp21);
                    auto tmp30 = decltype(tmp27)(tmp27 - tmp29);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp31);
                    auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                    in_out_ptr0[static_cast<long>(x1 + (40L*x0))] = tmp33;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_native_batch_norm_backward_threshold_backward_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (192L*x1))];
                        auto tmp26 = in_ptr3[static_cast<long>(x0 + (192L*x1))];
                        auto tmp27 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = c10::convert<long>(x0);
                        auto tmp2 = static_cast<long>(0);
                        auto tmp3 = tmp1 >= tmp2;
                        auto tmp4 = static_cast<long>(96);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = in_ptr1[static_cast<long>(x0 + (96L*x1))];
                            return tmp7;
                        }
                        ;
                        auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                        auto tmp9 = tmp1 >= tmp4;
                        auto tmp10 = static_cast<long>(192);
                        auto tmp11 = tmp1 < tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr2[static_cast<long>((-96L) + x0 + (96L*x1))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp5 ? tmp8 : tmp14;
                        auto tmp16 = static_cast<float>(0.0);
                        auto tmp17 = tmp0 ? tmp16 : tmp15;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr1[static_cast<long>(x0 + (96L*x1))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp5 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = [&]
                        {
                            auto tmp22 = in_ptr2[static_cast<long>((-96L) + x0 + (96L*x1))];
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp9 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                        auto tmp24 = tmp5 ? tmp20 : tmp23;
                        auto tmp25 = tmp0 ? tmp16 : tmp24;
                        auto tmp28 = decltype(tmp26)(tmp26 - tmp27);
                        auto tmp29 = decltype(tmp25)(tmp25 * tmp28);
                        tmp_acc0 = tmp_acc0 + tmp17;
                        tmp_acc1 = tmp_acc1 + tmp29;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                    auto tmp18 = in_ptr3[static_cast<long>(x1 + (192L*x0))];
                    auto tmp19 = in_ptr4[static_cast<long>(x1)];
                    auto tmp21 = out_ptr1[static_cast<long>(x1)];
                    auto tmp24 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = out_ptr0[static_cast<long>(x1)];
                    auto tmp32 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = c10::convert<long>(x1);
                    auto tmp2 = static_cast<long>(0);
                    auto tmp3 = tmp1 >= tmp2;
                    auto tmp4 = static_cast<long>(96);
                    auto tmp5 = tmp1 < tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = in_ptr1[static_cast<long>(x1 + (96L*x0))];
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                    auto tmp9 = tmp1 >= tmp4;
                    auto tmp10 = static_cast<long>(192);
                    auto tmp11 = tmp1 < tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr2[static_cast<long>((-96L) + x1 + (96L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp9 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp5 ? tmp8 : tmp14;
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp0 ? tmp16 : tmp15;
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    auto tmp22 = static_cast<float>(3.985969387755102e-05);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp25 = decltype(tmp24)(tmp24 * tmp24);
                    auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                    auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                    auto tmp28 = decltype(tmp17)(tmp17 - tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp22);
                    auto tmp31 = decltype(tmp28)(tmp28 - tmp30);
                    auto tmp33 = decltype(tmp24)(tmp24 * tmp32);
                    auto tmp34 = decltype(tmp31)(tmp31 * tmp33);
                    out_ptr2[static_cast<long>(x1 + (192L*x0))] = tmp34;
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_batch_norm_backward_threshold_backward_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(112L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(64);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x1);
                                auto tmp7 = static_cast<long>(113);
                                auto tmp8 = tmp6 < tmp7;
                                auto tmp9 = c10::convert<long>(x2);
                                auto tmp10 = tmp9 < tmp7;
                                auto tmp11 = tmp8 & tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr0[static_cast<long>(x3 + (64L*x2) + (7232L*x1) + (817216L*x0))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp16 = tmp0 >= tmp3;
                            auto tmp17 = static_cast<long>(128);
                            auto tmp18 = tmp0 < tmp17;
                            auto tmp19 = tmp16 & tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(1L + x1);
                                auto tmp22 = static_cast<long>(0);
                                auto tmp23 = tmp21 >= tmp22;
                                auto tmp24 = static_cast<long>(115);
                                auto tmp25 = tmp21 < tmp24;
                                auto tmp26 = c10::convert<long>(1L + x2);
                                auto tmp27 = tmp26 >= tmp22;
                                auto tmp28 = tmp26 < tmp24;
                                auto tmp29 = tmp23 & tmp25;
                                auto tmp30 = tmp29 & tmp27;
                                auto tmp31 = tmp30 & tmp28;
                                auto tmp32 = [&]
                                {
                                    auto tmp33 = in_ptr1[static_cast<long>(7360L + x3 + (64L*x2) + (7360L*x1) + (846400L*x0))];
                                    return tmp33;
                                }
                                ;
                                auto tmp34 = tmp31 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp36 = tmp0 >= tmp17;
                            auto tmp37 = static_cast<long>(192);
                            auto tmp38 = tmp0 < tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(2L + x1);
                                auto tmp41 = static_cast<long>(0);
                                auto tmp42 = tmp40 >= tmp41;
                                auto tmp43 = static_cast<long>(117);
                                auto tmp44 = tmp40 < tmp43;
                                auto tmp45 = c10::convert<long>(2L + x2);
                                auto tmp46 = tmp45 >= tmp41;
                                auto tmp47 = tmp45 < tmp43;
                                auto tmp48 = tmp42 & tmp44;
                                auto tmp49 = tmp48 & tmp46;
                                auto tmp50 = tmp49 & tmp47;
                                auto tmp51 = [&]
                                {
                                    auto tmp52 = in_ptr2[static_cast<long>(14976L + x3 + (64L*x2) + (7488L*x1) + (876096L*x0))];
                                    return tmp52;
                                }
                                ;
                                auto tmp53 = tmp50 ? tmp51() : static_cast<decltype(tmp51())>(0.0);
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp36 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp55 = tmp19 ? tmp35 : tmp54;
                            auto tmp56 = tmp4 ? tmp15 : tmp55;
                            out_ptr0[static_cast<long>(x3 + (192L*x2) + (21504L*x1) + (2408448L*x0))] = tmp56;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_native_batch_norm_backward_98 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(1L))
                        {
                            auto tmp22 = in_ptr2[static_cast<long>(x0 + (32L*x2) + (401408L*x1))];
                            auto tmp23 = in_ptr3[static_cast<long>(x0)];
                            auto tmp0 = c10::convert<long>(x0);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(16);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2 + (12544L*x0) + (200704L*x1))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(32);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>((-200704L) + x2 + (12544L*x0) + (200704L*x1))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = tmp4 ? tmp7 : tmp13;
                            auto tmp15 = [&]
                            {
                                auto tmp16 = in_ptr0[static_cast<long>(x2 + (12544L*x0) + (200704L*x1))];
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp4 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = in_ptr1[static_cast<long>((-200704L) + x2 + (12544L*x0) + (200704L*x1))];
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp8 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                            auto tmp21 = tmp4 ? tmp17 : tmp20;
                            auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                            auto tmp25 = decltype(tmp21)(tmp21 * tmp24);
                            tmp_acc0 = tmp_acc0 + tmp14;
                            tmp_acc1 = tmp_acc1 + tmp25;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(1L))
                    {
                        auto tmp15 = in_ptr2[static_cast<long>(x1 + (32L*x2) + (401408L*x0))];
                        auto tmp16 = in_ptr3[static_cast<long>(x1)];
                        auto tmp18 = out_ptr1[static_cast<long>(x1)];
                        auto tmp21 = in_ptr4[static_cast<long>(x1)];
                        auto tmp26 = out_ptr0[static_cast<long>(x1)];
                        auto tmp29 = in_ptr5[static_cast<long>(x1)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (12544L*x1) + (200704L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(32);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-200704L) + x2 + (12544L*x1) + (200704L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        auto tmp17 = decltype(tmp15)(tmp15 - tmp16);
                        auto tmp19 = static_cast<float>(9.964923469387754e-06);
                        auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                        auto tmp22 = decltype(tmp21)(tmp21 * tmp21);
                        auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                        auto tmp24 = decltype(tmp17)(tmp17 * tmp23);
                        auto tmp25 = decltype(tmp14)(tmp14 - tmp24);
                        auto tmp27 = decltype(tmp26)(tmp26 * tmp19);
                        auto tmp28 = decltype(tmp25)(tmp25 - tmp27);
                        auto tmp30 = decltype(tmp21)(tmp21 * tmp29);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        out_ptr3[static_cast<long>(x2 + (12544L*x1) + (401408L*x0))] = tmp31;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_convolution_backward_native_batch_norm_backward_threshold_backward_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr3 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (32L*x2) + (401408L*x1))];
                            auto tmp18 = in_ptr3[static_cast<long>(x0 + (32L*x2) + (401408L*x1))];
                            auto tmp30 = in_ptr4[static_cast<long>(x0 + (32L*x2) + (401408L*x1))];
                            auto tmp31 = in_ptr5[static_cast<long>(x0)];
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = tmp0 <= tmp1;
                            auto tmp3 = c10::convert<long>(x0);
                            auto tmp4 = static_cast<long>(0);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = static_cast<long>(16);
                            auto tmp7 = tmp3 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr1[static_cast<long>(x2 + (12544L*x0) + (200704L*x1))];
                                return tmp9;
                            }
                            ;
                            auto tmp10 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp11 = tmp3 >= tmp6;
                            auto tmp12 = static_cast<long>(32);
                            auto tmp13 = tmp3 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr2[static_cast<long>((-200704L) + x2 + (12544L*x0) + (200704L*x1))];
                                return tmp15;
                            }
                            ;
                            auto tmp16 = tmp11 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp17 = tmp7 ? tmp10 : tmp16;
                            auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                            auto tmp20 = tmp2 ? tmp1 : tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (12544L*x0) + (200704L*x1))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp7 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = [&]
                            {
                                auto tmp25 = in_ptr2[static_cast<long>((-200704L) + x2 + (12544L*x0) + (200704L*x1))];
                                return tmp25;
                            }
                            ;
                            auto tmp26 = tmp11 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                            auto tmp27 = tmp7 ? tmp23 : tmp26;
                            auto tmp28 = decltype(tmp27)(tmp27 + tmp18);
                            auto tmp29 = tmp2 ? tmp1 : tmp28;
                            auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                            auto tmp33 = decltype(tmp29)(tmp29 * tmp32);
                            tmp_acc0 = tmp_acc0 + tmp20;
                            tmp_acc1 = tmp_acc1 + tmp33;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (32L*x1) + (401408L*x0))];
                        auto tmp18 = in_out_ptr0[static_cast<long>(x2 + (32L*x1) + (401408L*x0))];
                        auto tmp21 = in_ptr4[static_cast<long>(x2 + (32L*x1) + (401408L*x0))];
                        auto tmp22 = in_ptr5[static_cast<long>(x2)];
                        auto tmp24 = out_ptr1[static_cast<long>(x2)];
                        auto tmp27 = in_ptr6[static_cast<long>(x2)];
                        auto tmp32 = out_ptr0[static_cast<long>(x2)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp3 = c10::convert<long>(x2);
                        auto tmp4 = static_cast<long>(0);
                        auto tmp5 = tmp3 >= tmp4;
                        auto tmp6 = static_cast<long>(16);
                        auto tmp7 = tmp3 < tmp6;
                        auto tmp8 = [&]
                        {
                            auto tmp9 = in_ptr1[static_cast<long>(x1 + (12544L*x2) + (200704L*x0))];
                            return tmp9;
                        }
                        ;
                        auto tmp10 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                        auto tmp11 = tmp3 >= tmp6;
                        auto tmp12 = static_cast<long>(32);
                        auto tmp13 = tmp3 < tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-200704L) + x1 + (12544L*x2) + (200704L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp11 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp7 ? tmp10 : tmp16;
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = tmp2 ? tmp1 : tmp19;
                        auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                        auto tmp25 = static_cast<float>(9.964923469387754e-06);
                        auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp27);
                        auto tmp29 = decltype(tmp26)(tmp26 * tmp28);
                        auto tmp30 = decltype(tmp23)(tmp23 * tmp29);
                        auto tmp31 = decltype(tmp20)(tmp20 - tmp30);
                        auto tmp33 = decltype(tmp32)(tmp32 * tmp25);
                        auto tmp34 = decltype(tmp31)(tmp31 - tmp33);
                        in_out_ptr0[static_cast<long>(x2 + (32L*x1) + (401408L*x0))] = tmp34;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_12, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_54, primals_55, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_236, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, constant_pad_nd, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, getitem_6, getitem_7, cat, squeeze_10, constant_pad_nd_1, constant_pad_nd_2, constant_pad_nd_3, cat_1, squeeze_13, getitem_26, getitem_29, cat_2, squeeze_16, getitem_32, getitem_33, cat_3, squeeze_19, relu_4, convolution_12, squeeze_22, getitem_40, getitem_43, cat_4, squeeze_25, add_46, convolution_15, squeeze_28, constant_pad_nd_4, constant_pad_nd_5, constant_pad_nd_6, constant_pad_nd_7, cat_5, squeeze_31, add_56, mean, convolution_20, mul_79, convolution_21, mul_80, convolution_22, squeeze_34, getitem_72, getitem_73, cat_6, squeeze_37, getitem_78, getitem_81, cat_7, squeeze_40, add_71, mean_1, convolution_27, mul_104, convolution_28, getitem_84, getitem_85, cat_8, squeeze_43, getitem_88, getitem_89, cat_9, squeeze_46, getitem_94, getitem_97, cat_10, squeeze_49, add_87, mean_2, convolution_35, mul_129, convolution_36, getitem_100, getitem_101, cat_11, squeeze_52, getitem_104, getitem_105, cat_12, squeeze_55, getitem_110, getitem_113, cat_13, squeeze_58, add_103, mean_3, convolution_43, mul_154, convolution_44, getitem_116, getitem_117, cat_14, squeeze_61, add_109, convolution_47, squeeze_64, constant_pad_nd_8, constant_pad_nd_9, constant_pad_nd_10, cat_15, squeeze_67, add_119, mean_4, convolution_51, mul_179, convolution_52, mul_180, convolution_53, squeeze_70, getitem_138, getitem_139, cat_16, squeeze_73, getitem_146, getitem_151, getitem_156, getitem_161, cat_17, squeeze_76, add_134, mean_5, convolution_60, mul_204, convolution_61, getitem_164, getitem_165, cat_18, squeeze_79, getitem_168, getitem_169, cat_19, squeeze_82, getitem_176, getitem_181, getitem_186, getitem_191, cat_20, squeeze_85, add_150, mean_6, convolution_70, mul_229, convolution_71, getitem_194, getitem_195, cat_21, squeeze_88, getitem_198, getitem_199, cat_22, squeeze_91, getitem_206, getitem_211, getitem_216, getitem_221, cat_23, squeeze_94, add_166, mean_7, convolution_80, mul_254, convolution_81, getitem_224, getitem_225, cat_24, squeeze_97, add_172, convolution_84, squeeze_100, mul_270, convolution_85, squeeze_103, add_182, mean_8, convolution_86, mul_279, convolution_87, mul_280, convolution_88, squeeze_106, getitem_234, getitem_235, cat_25, squeeze_109, getitem_242, getitem_247, getitem_252, getitem_257, cat_26, squeeze_112, add_197, mean_9, convolution_95, mul_304, convolution_96, getitem_260, getitem_261, cat_27, squeeze_115, getitem_264, getitem_265, cat_28, squeeze_118, getitem_272, getitem_277, getitem_282, getitem_287, cat_29, squeeze_121, add_213, mean_10, convolution_105, mul_329, convolution_106, getitem_290, getitem_291, cat_30, squeeze_124, getitem_294, getitem_295, cat_31, squeeze_127, getitem_302, getitem_307, getitem_312, getitem_317, cat_32, squeeze_130, add_229, mean_11, convolution_115, mul_354, convolution_116, getitem_320, getitem_321, cat_33, squeeze_133, add_235, convolution_119, squeeze_136, constant_pad_nd_11, constant_pad_nd_12, constant_pad_nd_13, constant_pad_nd_14, cat_34, squeeze_139, add_245, mean_12, convolution_124, mul_379, convolution_125, mul_380, convolution_126, squeeze_142, add_250, convolution_127, squeeze_145, getitem_356, getitem_361, getitem_366, getitem_371, cat_35, squeeze_148, add_260, mean_13, convolution_132, mul_404, convolution_133, getitem_374, getitem_375, cat_36, squeeze_151, add_266, convolution_136, squeeze_154, getitem_384, getitem_389, getitem_394, getitem_399, cat_37, squeeze_157, add_276, mean_14, convolution_141, mul_429, convolution_142, getitem_402, getitem_403, cat_38, squeeze_160, add_282, convolution_145, squeeze_163, getitem_412, getitem_417, getitem_422, getitem_427, cat_39, squeeze_166, add_292, mean_15, convolution_150, mul_454, convolution_151, getitem_430, getitem_431, cat_40, squeeze_169, add_298, convolution_154, squeeze_172, view, permute_1, le, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_508, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_548, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_588, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_628, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_668, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_708, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_748, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_788, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_828, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_868, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_908, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_948, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_988, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1028, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1068, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1108, unsqueeze_810, unsqueeze_822, le_1, unsqueeze_834, unsqueeze_846, unsqueeze_858, le_3, unsqueeze_870, le_4, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_8, (192, ), (1, ))
    assert_size_stride(primals_10, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_11, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_12, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_15, (40, ), (1, ))
    assert_size_stride(primals_17, (120, ), (1, ))
    assert_size_stride(primals_19, (120, ), (1, ))
    assert_size_stride(primals_21, (40, ), (1, ))
    assert_size_stride(primals_23, (240, ), (1, ))
    assert_size_stride(primals_25, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_26, (60, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_27, (60, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_28, (60, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_29, (240, ), (1, ))
    assert_size_stride(primals_31, (56, ), (1, ))
    assert_size_stride(primals_33, (336, ), (1, ))
    assert_size_stride(primals_35, (336, ), (1, ))
    assert_size_stride(primals_37, (56, ), (1, ))
    assert_size_stride(primals_39, (336, ), (1, ))
    assert_size_stride(primals_41, (336, ), (1, ))
    assert_size_stride(primals_43, (56, ), (1, ))
    assert_size_stride(primals_45, (336, ), (1, ))
    assert_size_stride(primals_47, (336, ), (1, ))
    assert_size_stride(primals_49, (56, ), (1, ))
    assert_size_stride(primals_51, (336, ), (1, ))
    assert_size_stride(primals_53, (112, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_54, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_55, (112, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_56, (336, ), (1, ))
    assert_size_stride(primals_58, (104, ), (1, ))
    assert_size_stride(primals_60, (624, ), (1, ))
    assert_size_stride(primals_62, (624, ), (1, ))
    assert_size_stride(primals_64, (104, ), (1, ))
    assert_size_stride(primals_66, (624, ), (1, ))
    assert_size_stride(primals_68, (624, ), (1, ))
    assert_size_stride(primals_70, (104, ), (1, ))
    assert_size_stride(primals_72, (624, ), (1, ))
    assert_size_stride(primals_74, (624, ), (1, ))
    assert_size_stride(primals_76, (104, ), (1, ))
    assert_size_stride(primals_78, (624, ), (1, ))
    assert_size_stride(primals_80, (624, ), (1, ))
    assert_size_stride(primals_82, (160, ), (1, ))
    assert_size_stride(primals_84, (480, ), (1, ))
    assert_size_stride(primals_86, (480, ), (1, ))
    assert_size_stride(primals_88, (160, ), (1, ))
    assert_size_stride(primals_90, (480, ), (1, ))
    assert_size_stride(primals_92, (480, ), (1, ))
    assert_size_stride(primals_94, (160, ), (1, ))
    assert_size_stride(primals_96, (480, ), (1, ))
    assert_size_stride(primals_98, (480, ), (1, ))
    assert_size_stride(primals_100, (160, ), (1, ))
    assert_size_stride(primals_102, (960, ), (1, ))
    assert_size_stride(primals_104, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_106, (240, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_107, (240, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_108, (960, ), (1, ))
    assert_size_stride(primals_110, (264, ), (1, ))
    assert_size_stride(primals_112, (1584, ), (1, ))
    assert_size_stride(primals_114, (1584, ), (1, ))
    assert_size_stride(primals_116, (264, ), (1, ))
    assert_size_stride(primals_118, (1584, ), (1, ))
    assert_size_stride(primals_120, (1584, ), (1, ))
    assert_size_stride(primals_122, (264, ), (1, ))
    assert_size_stride(primals_124, (1584, ), (1, ))
    assert_size_stride(primals_126, (1584, ), (1, ))
    assert_size_stride(primals_128, (264, ), (1, ))
    assert_size_stride(primals_130, (1536, ), (1, ))
    assert_size_stride(primals_132, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_134, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_135, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_136, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_137, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_138, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_139, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_140, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_141, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_142, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_143, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_144, (20, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_146, (240, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_148, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_149, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_150, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_151, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_153, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_155, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_157, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_158, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_159, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_160, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_161, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_162, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_163, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_165, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_167, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_168, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_169, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_170, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_171, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_172, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_173, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_175, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_177, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_178, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_179, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_180, (14, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_182, (336, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(primals_184, (104, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_185, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_186, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_187, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_188, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_189, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_190, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_191, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_193, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_195, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_196, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_197, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_198, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_199, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_200, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_201, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_202, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_203, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_205, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_207, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_208, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_209, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_210, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_211, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_212, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_213, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_214, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_215, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_217, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_219, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_220, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_221, (624, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_222, (624, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_223, (52, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_225, (624, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_227, (160, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_228, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_229, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_230, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_231, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_232, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_233, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_234, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_236, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_238, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_239, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_240, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_241, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_242, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_244, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_245, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_246, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_248, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_250, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_251, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_252, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_253, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_254, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_255, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_256, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_257, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_258, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_260, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_262, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_263, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_264, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_265, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_267, (960, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_269, (264, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_270, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_271, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_272, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_273, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_274, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_275, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(primals_277, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_279, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_280, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_281, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_282, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_283, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_284, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_285, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_286, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(primals_288, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_290, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_291, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_292, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_293, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_294, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_295, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_296, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_297, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(primals_299, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_301, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_302, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_303, (1536, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(constant_pad_nd, (8, 3, 225, 225), (151875, 1, 675, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(relu_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_7, (32, ), (1, ))
    assert_size_stride(getitem_6, (8, 16, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(getitem_7, (8, 16, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(cat, (8, 192, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(squeeze_10, (192, ), (1, ))
    assert_size_stride(constant_pad_nd_1, (8, 64, 113, 113), (817216, 1, 7232, 64))
    assert_size_stride(constant_pad_nd_2, (8, 64, 115, 115), (846400, 1, 7360, 64))
    assert_size_stride(constant_pad_nd_3, (8, 64, 117, 117), (876096, 1, 7488, 64))
    assert_size_stride(cat_1, (8, 192, 56, 56), (602112, 1, 10752, 192))
    assert_size_stride(squeeze_13, (192, ), (1, ))
    assert_size_stride(getitem_26, (8, 96, 56, 56), (602112, 1, 10752, 192))
    assert_size_stride(getitem_29, (8, 96, 56, 56), (602112, 1, 10752, 192))
    assert_size_stride(cat_2, (8, 40, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(squeeze_16, (40, ), (1, ))
    assert_size_stride(getitem_32, (8, 20, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(getitem_33, (8, 20, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(cat_3, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(squeeze_19, (120, ), (1, ))
    assert_size_stride(relu_4, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(convolution_12, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(squeeze_22, (120, ), (1, ))
    assert_size_stride(getitem_40, (8, 60, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(getitem_43, (8, 60, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(cat_4, (8, 40, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(squeeze_25, (40, ), (1, ))
    assert_size_stride(add_46, (8, 40, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(convolution_15, (8, 240, 56, 56), (752640, 1, 13440, 240))
    assert_size_stride(squeeze_28, (240, ), (1, ))
    assert_size_stride(constant_pad_nd_4, (8, 60, 57, 57), (194940, 1, 3420, 60))
    assert_size_stride(constant_pad_nd_5, (8, 60, 59, 59), (208860, 1, 3540, 60))
    assert_size_stride(constant_pad_nd_6, (8, 60, 61, 61), (223260, 1, 3660, 60))
    assert_size_stride(constant_pad_nd_7, (8, 60, 63, 63), (238140, 1, 3780, 60))
    assert_size_stride(cat_5, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_31, (240, ), (1, ))
    assert_size_stride(add_56, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(mean, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_20, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_79, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_21, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_80, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_22, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_34, (56, ), (1, ))
    assert_size_stride(getitem_72, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(getitem_73, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(cat_6, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_37, (336, ), (1, ))
    assert_size_stride(getitem_78, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_81, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_7, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_40, (336, ), (1, ))
    assert_size_stride(add_71, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(mean_1, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(convolution_27, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_104, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_28, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(getitem_84, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_85, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_8, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_43, (56, ), (1, ))
    assert_size_stride(getitem_88, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(getitem_89, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(cat_9, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_46, (336, ), (1, ))
    assert_size_stride(getitem_94, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_97, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_10, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_49, (336, ), (1, ))
    assert_size_stride(add_87, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(mean_2, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(convolution_35, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_129, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_36, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(getitem_100, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_101, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_11, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_52, (56, ), (1, ))
    assert_size_stride(getitem_104, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(getitem_105, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(cat_12, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_55, (336, ), (1, ))
    assert_size_stride(getitem_110, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_113, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_13, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_58, (336, ), (1, ))
    assert_size_stride(add_103, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(mean_3, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(convolution_43, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_154, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_44, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(getitem_116, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_117, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_14, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_61, (56, ), (1, ))
    assert_size_stride(add_109, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(convolution_47, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_64, (336, ), (1, ))
    assert_size_stride(constant_pad_nd_8, (8, 112, 29, 29), (94192, 1, 3248, 112))
    assert_size_stride(constant_pad_nd_9, (8, 112, 31, 31), (107632, 1, 3472, 112))
    assert_size_stride(constant_pad_nd_10, (8, 112, 33, 33), (121968, 1, 3696, 112))
    assert_size_stride(cat_15, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(squeeze_67, (336, ), (1, ))
    assert_size_stride(add_119, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(mean_4, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(convolution_51, (8, 14, 1, 1), (14, 1, 14, 14))
    assert_size_stride(mul_179, (8, 14, 1, 1), (14, 1, 14, 14))
    assert_size_stride(convolution_52, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(mul_180, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(convolution_53, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_70, (104, ), (1, ))
    assert_size_stride(getitem_138, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(getitem_139, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(cat_16, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_73, (624, ), (1, ))
    assert_size_stride(getitem_146, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_151, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_156, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_161, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_17, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_76, (624, ), (1, ))
    assert_size_stride(add_134, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(mean_5, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(convolution_60, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(mul_204, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(convolution_61, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(getitem_164, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_165, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_18, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_79, (104, ), (1, ))
    assert_size_stride(getitem_168, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(getitem_169, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(cat_19, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_82, (624, ), (1, ))
    assert_size_stride(getitem_176, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_181, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_186, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_191, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_20, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_85, (624, ), (1, ))
    assert_size_stride(add_150, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(mean_6, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(convolution_70, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(mul_229, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(convolution_71, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(getitem_194, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_195, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_21, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_88, (104, ), (1, ))
    assert_size_stride(getitem_198, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(getitem_199, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(cat_22, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_91, (624, ), (1, ))
    assert_size_stride(getitem_206, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_211, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_216, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_221, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_23, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_94, (624, ), (1, ))
    assert_size_stride(add_166, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(mean_7, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(convolution_80, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(mul_254, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(convolution_81, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(getitem_224, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_225, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_24, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_97, (104, ), (1, ))
    assert_size_stride(add_172, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_84, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_100, (624, ), (1, ))
    assert_size_stride(mul_270, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(convolution_85, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_103, (624, ), (1, ))
    assert_size_stride(add_182, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(mean_8, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(convolution_86, (8, 52, 1, 1), (52, 1, 52, 52))
    assert_size_stride(mul_279, (8, 52, 1, 1), (52, 1, 52, 52))
    assert_size_stride(convolution_87, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(mul_280, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(convolution_88, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(squeeze_106, (160, ), (1, ))
    assert_size_stride(getitem_234, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(getitem_235, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(cat_25, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_109, (480, ), (1, ))
    assert_size_stride(getitem_242, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_247, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_252, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_257, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_26, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_112, (480, ), (1, ))
    assert_size_stride(add_197, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_9, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_95, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(mul_304, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(convolution_96, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(getitem_260, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_261, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_27, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(squeeze_115, (160, ), (1, ))
    assert_size_stride(getitem_264, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(getitem_265, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(cat_28, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_118, (480, ), (1, ))
    assert_size_stride(getitem_272, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_277, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_282, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_287, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_29, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_121, (480, ), (1, ))
    assert_size_stride(add_213, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_10, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_105, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(mul_329, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(convolution_106, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(getitem_290, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_291, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_30, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(squeeze_124, (160, ), (1, ))
    assert_size_stride(getitem_294, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(getitem_295, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(cat_31, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_127, (480, ), (1, ))
    assert_size_stride(getitem_302, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_307, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_312, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_317, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_32, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_130, (480, ), (1, ))
    assert_size_stride(add_229, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_11, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_115, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(mul_354, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(convolution_116, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(getitem_320, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_321, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_33, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(squeeze_133, (160, ), (1, ))
    assert_size_stride(add_235, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(convolution_119, (8, 960, 14, 14), (188160, 1, 13440, 960))
    assert_size_stride(squeeze_136, (960, ), (1, ))
    assert_size_stride(constant_pad_nd_11, (8, 240, 15, 15), (54000, 1, 3600, 240))
    assert_size_stride(constant_pad_nd_12, (8, 240, 17, 17), (69360, 1, 4080, 240))
    assert_size_stride(constant_pad_nd_13, (8, 240, 19, 19), (86640, 1, 4560, 240))
    assert_size_stride(constant_pad_nd_14, (8, 240, 21, 21), (105840, 1, 5040, 240))
    assert_size_stride(cat_34, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_139, (960, ), (1, ))
    assert_size_stride(add_245, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_12, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(convolution_124, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(mul_379, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(convolution_125, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_380, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_126, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(squeeze_142, (264, ), (1, ))
    assert_size_stride(add_250, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(convolution_127, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_145, (1584, ), (1, ))
    assert_size_stride(getitem_356, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_361, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_366, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_371, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_35, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_148, (1584, ), (1, ))
    assert_size_stride(add_260, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(mean_13, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(convolution_132, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(mul_404, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(convolution_133, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(getitem_374, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_375, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_36, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(squeeze_151, (264, ), (1, ))
    assert_size_stride(add_266, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(convolution_136, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_154, (1584, ), (1, ))
    assert_size_stride(getitem_384, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_389, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_394, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_399, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_37, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_157, (1584, ), (1, ))
    assert_size_stride(add_276, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(mean_14, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(convolution_141, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(mul_429, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(convolution_142, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(getitem_402, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_403, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_38, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(squeeze_160, (264, ), (1, ))
    assert_size_stride(add_282, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(convolution_145, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_163, (1584, ), (1, ))
    assert_size_stride(getitem_412, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_417, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_422, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_427, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_39, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_166, (1584, ), (1, ))
    assert_size_stride(add_292, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(mean_15, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(convolution_150, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(mul_454, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(convolution_151, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(getitem_430, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_431, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_40, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(squeeze_169, (264, ), (1, ))
    assert_size_stride(add_298, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(convolution_154, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(squeeze_172, (1536, ), (1, ))
    assert_size_stride(view, (8, 1536), (1536, 1))
    assert_size_stride(permute_1, (1000, 1536), (1536, 1))
    assert_size_stride(le, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(unsqueeze_234, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(unsqueeze_246, (1, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(mul_508, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(unsqueeze_270, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(mul_548, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(unsqueeze_306, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(unsqueeze_318, (1, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(mul_588, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(unsqueeze_342, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(mul_628, (8, 960, 14, 14), (188160, 1, 13440, 960))
    assert_size_stride(unsqueeze_378, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_668, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_414, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_708, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_450, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_748, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_486, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(mul_788, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(unsqueeze_522, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(mul_828, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(unsqueeze_558, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(mul_868, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(unsqueeze_594, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(mul_908, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(unsqueeze_630, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(mul_948, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(unsqueeze_666, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(mul_988, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(unsqueeze_702, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(mul_1028, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(unsqueeze_738, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(mul_1068, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(unsqueeze_774, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_786, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_798, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_1108, (8, 240, 56, 56), (752640, 1, 13440, 240))
    assert_size_stride(unsqueeze_810, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_822, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(le_1, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(unsqueeze_834, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_846, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_858, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(le_3, (8, 192, 56, 56), (602112, 1, 10752, 192))
    assert_size_stride(unsqueeze_870, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_4, (8, 192, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(unsqueeze_882, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_894, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_906, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_918, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view, out=buf1)
    del view
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf4 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf5 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_154.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del buf4
    del convolution_154
    del le
    del primals_130
    del squeeze_172
    del tangents_1
    del unsqueeze_234
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, add_298, primals_303, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_298
    del buf6
    del primals_303
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((264, ), device='cpu', dtype=torch.float32)
    buf11 = empty((264, ), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cpu', dtype=torch.float32)
    buf13 = buf11; del buf11  # reuse
    cpp_fused_native_batch_norm_backward_1(c_void_p(buf13.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(cat_40.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()))
    del cat_40
    del primals_128
    del squeeze_169
    del unsqueeze_246
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf14 = aten.convolution_backward(reinterpret_tensor(buf12, (8, 132, 7, 7), (12936, 1, 1848, 264), 132), getitem_431, primals_302, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_431
    del primals_302
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf17 = aten.convolution_backward(reinterpret_tensor(buf12, (8, 132, 7, 7), (12936, 1, 1848, 264), 0), getitem_430, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_430
    del primals_301
    buf18 = buf17[0]
    buf19 = buf17[1]
    del buf17
    buf20 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cpu', dtype=torch.float32)
    buf21 = reinterpret_tensor(buf20, (8, 1584, 1, 1), (1584, 1, 1, 1), 0); del buf20  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_2(c_void_p(buf21.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(add_292.data_ptr()), c_void_p(convolution_151.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf22 = aten.convolution_backward(buf21, mul_454, primals_299, [1584], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf21
    del mul_454
    del primals_299
    buf23 = buf22[0]
    buf24 = buf22[1]
    buf25 = buf22[2]
    del buf22
    buf26 = reinterpret_tensor(buf23, (8, 132, 1, 1), (132, 1, 1, 1), 0); del buf23  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_3(c_void_p(buf26.data_ptr()), c_void_p(convolution_150.data_ptr()))
    del convolution_150
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf27 = aten.convolution_backward(buf26, mean_15, primals_297, [132], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf26
    del mean_15
    del primals_297
    buf28 = buf27[0]
    buf29 = buf27[1]
    buf30 = buf27[2]
    del buf27
    buf31 = empty((8, 1584, 7, 7), device='cpu', dtype=torch.float32)
    buf32 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf33 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf34 = buf31; del buf31  # reuse
    buf35 = buf33; del buf33  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_4(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(convolution_151.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(add_292.data_ptr()), c_void_p(cat_39.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf32.data_ptr()))
    del add_292
    del buf15
    del buf18
    del cat_39
    del convolution_151
    del primals_126
    del squeeze_166
    del unsqueeze_258
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf36 = aten.convolution_backward(reinterpret_tensor(buf34, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), getitem_427, primals_296, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_427
    del primals_296
    buf37 = buf36[0]
    buf38 = buf36[1]
    del buf36
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf39 = aten.convolution_backward(reinterpret_tensor(buf34, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), getitem_422, primals_295, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_422
    del primals_295
    buf40 = buf39[0]
    buf41 = buf39[1]
    del buf39
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf42 = aten.convolution_backward(reinterpret_tensor(buf34, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), getitem_417, primals_294, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_417
    del primals_294
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf45 = aten.convolution_backward(reinterpret_tensor(buf34, (8, 396, 7, 7), (77616, 49, 7, 1), 0), getitem_412, primals_293, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_412
    del primals_293
    buf46 = buf45[0]
    buf47 = buf45[1]
    del buf45
    buf48 = buf34; del buf34  # reuse
    buf49 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf50 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf51 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf52 = buf48; del buf48  # reuse
    cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_5(c_void_p(buf52.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(mul_508.data_ptr()), c_void_p(convolution_145.data_ptr()), c_void_p(unsqueeze_270.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del buf37
    del buf40
    del buf43
    del buf46
    del convolution_145
    del mul_508
    del primals_124
    del squeeze_163
    del unsqueeze_270
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf53 = aten.convolution_backward(buf52, add_282, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_282
    del primals_292
    buf54 = buf53[0]
    buf55 = buf53[1]
    del buf53
    buf56 = empty((264, ), device='cpu', dtype=torch.float32)
    buf57 = empty((264, ), device='cpu', dtype=torch.float32)
    buf58 = buf12; del buf12  # reuse
    buf59 = buf57; del buf57  # reuse
    cpp_fused_add_native_batch_norm_backward_6(c_void_p(buf59.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(cat_38.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()))
    del cat_38
    del primals_122
    del squeeze_160
    del unsqueeze_282
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf60 = aten.convolution_backward(reinterpret_tensor(buf58, (8, 132, 7, 7), (12936, 1, 1848, 264), 132), getitem_403, primals_291, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_403
    del primals_291
    buf61 = buf60[0]
    buf62 = buf60[1]
    del buf60
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf63 = aten.convolution_backward(reinterpret_tensor(buf58, (8, 132, 7, 7), (12936, 1, 1848, 264), 0), getitem_402, primals_290, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_402
    del primals_290
    buf64 = buf63[0]
    buf65 = buf63[1]
    del buf63
    buf66 = reinterpret_tensor(buf28, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf28  # reuse
    buf67 = reinterpret_tensor(buf66, (8, 1584, 1, 1), (1584, 1, 1, 1), 0); del buf66  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_7(c_void_p(buf67.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(add_276.data_ptr()), c_void_p(convolution_142.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf68 = aten.convolution_backward(buf67, mul_429, primals_288, [1584], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf67
    del mul_429
    del primals_288
    buf69 = buf68[0]
    buf70 = buf68[1]
    buf71 = buf68[2]
    del buf68
    buf72 = reinterpret_tensor(buf69, (8, 132, 1, 1), (132, 1, 1, 1), 0); del buf69  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_8(c_void_p(buf72.data_ptr()), c_void_p(convolution_141.data_ptr()))
    del convolution_141
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf73 = aten.convolution_backward(buf72, mean_14, primals_286, [132], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf72
    del mean_14
    del primals_286
    buf74 = buf73[0]
    buf75 = buf73[1]
    buf76 = buf73[2]
    del buf73
    buf77 = buf52; del buf52  # reuse
    buf78 = buf50; del buf50  # reuse
    buf79 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf80 = buf77; del buf77  # reuse
    buf81 = buf79; del buf79  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_9(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(convolution_142.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(add_276.data_ptr()), c_void_p(cat_37.data_ptr()), c_void_p(unsqueeze_294.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf78.data_ptr()))
    del add_276
    del buf61
    del buf64
    del cat_37
    del convolution_142
    del primals_120
    del squeeze_157
    del unsqueeze_294
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf82 = aten.convolution_backward(reinterpret_tensor(buf80, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), getitem_399, primals_285, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_399
    del primals_285
    buf83 = buf82[0]
    buf84 = buf82[1]
    del buf82
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf85 = aten.convolution_backward(reinterpret_tensor(buf80, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), getitem_394, primals_284, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_394
    del primals_284
    buf86 = buf85[0]
    buf87 = buf85[1]
    del buf85
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf88 = aten.convolution_backward(reinterpret_tensor(buf80, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), getitem_389, primals_283, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_389
    del primals_283
    buf89 = buf88[0]
    buf90 = buf88[1]
    del buf88
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf91 = aten.convolution_backward(reinterpret_tensor(buf80, (8, 396, 7, 7), (77616, 49, 7, 1), 0), getitem_384, primals_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_384
    del primals_282
    buf92 = buf91[0]
    buf93 = buf91[1]
    del buf91
    buf94 = buf80; del buf80  # reuse
    buf95 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf96 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf97 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf98 = buf94; del buf94  # reuse
    cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_10(c_void_p(buf98.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(mul_548.data_ptr()), c_void_p(convolution_136.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del buf83
    del buf86
    del buf89
    del buf92
    del convolution_136
    del mul_548
    del primals_118
    del squeeze_154
    del unsqueeze_306
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf99 = aten.convolution_backward(buf98, add_266, primals_281, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_266
    del primals_281
    buf100 = buf99[0]
    buf101 = buf99[1]
    del buf99
    buf102 = empty((264, ), device='cpu', dtype=torch.float32)
    buf103 = empty((264, ), device='cpu', dtype=torch.float32)
    buf104 = buf58; del buf58  # reuse
    buf105 = buf103; del buf103  # reuse
    cpp_fused_add_native_batch_norm_backward_11(c_void_p(buf105.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(cat_36.data_ptr()), c_void_p(unsqueeze_318.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()))
    del cat_36
    del primals_116
    del squeeze_151
    del unsqueeze_318
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf106 = aten.convolution_backward(reinterpret_tensor(buf104, (8, 132, 7, 7), (12936, 1, 1848, 264), 132), getitem_375, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_375
    del primals_280
    buf107 = buf106[0]
    buf108 = buf106[1]
    del buf106
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf109 = aten.convolution_backward(reinterpret_tensor(buf104, (8, 132, 7, 7), (12936, 1, 1848, 264), 0), getitem_374, primals_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf104
    del getitem_374
    del primals_279
    buf110 = buf109[0]
    buf111 = buf109[1]
    del buf109
    buf112 = reinterpret_tensor(buf74, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf74  # reuse
    buf113 = reinterpret_tensor(buf112, (8, 1584, 1, 1), (1584, 1, 1, 1), 0); del buf112  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_12(c_void_p(buf113.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(add_260.data_ptr()), c_void_p(convolution_133.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf114 = aten.convolution_backward(buf113, mul_404, primals_277, [1584], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf113
    del mul_404
    del primals_277
    buf115 = buf114[0]
    buf116 = buf114[1]
    buf117 = buf114[2]
    del buf114
    buf118 = reinterpret_tensor(buf115, (8, 132, 1, 1), (132, 1, 1, 1), 0); del buf115  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_13(c_void_p(buf118.data_ptr()), c_void_p(convolution_132.data_ptr()))
    del convolution_132
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf119 = aten.convolution_backward(buf118, mean_13, primals_275, [132], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf118
    del mean_13
    del primals_275
    buf120 = buf119[0]
    buf121 = buf119[1]
    buf122 = buf119[2]
    del buf119
    buf123 = buf98; del buf98  # reuse
    buf124 = buf96; del buf96  # reuse
    buf125 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf126 = buf123; del buf123  # reuse
    buf127 = buf125; del buf125  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14(c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(convolution_133.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(add_260.data_ptr()), c_void_p(cat_35.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf124.data_ptr()))
    del add_260
    del buf107
    del buf110
    del buf120
    del cat_35
    del convolution_133
    del primals_114
    del squeeze_148
    del unsqueeze_330
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf128 = aten.convolution_backward(reinterpret_tensor(buf126, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), getitem_371, primals_274, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_371
    del primals_274
    buf129 = buf128[0]
    buf130 = buf128[1]
    del buf128
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf131 = aten.convolution_backward(reinterpret_tensor(buf126, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), getitem_366, primals_273, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_366
    del primals_273
    buf132 = buf131[0]
    buf133 = buf131[1]
    del buf131
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf134 = aten.convolution_backward(reinterpret_tensor(buf126, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), getitem_361, primals_272, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_361
    del primals_272
    buf135 = buf134[0]
    buf136 = buf134[1]
    del buf134
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf137 = aten.convolution_backward(reinterpret_tensor(buf126, (8, 396, 7, 7), (77616, 49, 7, 1), 0), getitem_356, primals_271, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 396, [True, True, False])
    del getitem_356
    del primals_271
    buf138 = buf137[0]
    buf139 = buf137[1]
    del buf137
    buf140 = buf126; del buf126  # reuse
    buf141 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf142 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf143 = empty((1584, ), device='cpu', dtype=torch.float32)
    buf144 = buf140; del buf140  # reuse
    cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_15(c_void_p(buf144.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(mul_588.data_ptr()), c_void_p(convolution_127.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    del buf129
    del buf132
    del buf135
    del buf138
    del buf142
    del convolution_127
    del mul_588
    del primals_112
    del squeeze_145
    del unsqueeze_342
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf145 = aten.convolution_backward(buf144, add_250, primals_270, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_250
    del buf144
    del primals_270
    buf146 = buf145[0]
    buf147 = buf145[1]
    del buf145
    buf148 = empty((264, ), device='cpu', dtype=torch.float32)
    buf149 = empty((264, ), device='cpu', dtype=torch.float32)
    buf150 = buf100; del buf100  # reuse
    buf152 = buf150; del buf150  # reuse
    buf151 = buf149; del buf149  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_16(c_void_p(buf152.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(convolution_126.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf148.data_ptr()))
    del buf146
    del buf54
    del buf8
    del convolution_126
    del primals_110
    del squeeze_142
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf153 = aten.convolution_backward(buf152, mul_380, primals_269, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf152
    del mul_380
    del primals_269
    buf154 = buf153[0]
    buf155 = buf153[1]
    del buf153
    buf156 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cpu', dtype=torch.float32)
    buf157 = reinterpret_tensor(buf156, (8, 960, 1, 1), (960, 1, 1, 1), 0); del buf156  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_17(c_void_p(buf157.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(add_245.data_ptr()), c_void_p(convolution_125.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf158 = aten.convolution_backward(buf157, mul_379, primals_267, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf157
    del mul_379
    del primals_267
    buf159 = buf158[0]
    buf160 = buf158[1]
    buf161 = buf158[2]
    del buf158
    buf162 = reinterpret_tensor(buf159, (8, 80, 1, 1), (80, 1, 1, 1), 0); del buf159  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_18(c_void_p(buf162.data_ptr()), c_void_p(convolution_124.data_ptr()))
    del convolution_124
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf163 = aten.convolution_backward(buf162, mean_12, primals_265, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf162
    del mean_12
    del primals_265
    buf164 = buf163[0]
    buf165 = buf163[1]
    buf166 = buf163[2]
    del buf163
    buf167 = empty((960, ), device='cpu', dtype=torch.float32)
    buf168 = empty((960, ), device='cpu', dtype=torch.float32)
    buf169 = buf154; del buf154  # reuse
    buf170 = buf168; del buf168  # reuse
    buf171 = empty_strided((8, 240, 7, 7), (11760, 1, 1680, 240), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_19(c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(convolution_125.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(add_245.data_ptr()), c_void_p(cat_34.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf171.data_ptr()))
    del add_245
    del buf164
    del cat_34
    del convolution_125
    del unsqueeze_366
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf172 = aten.convolution_backward(buf171, constant_pad_nd_14, primals_107, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
    del constant_pad_nd_14
    del primals_107
    buf173 = buf172[0]
    buf174 = buf172[1]
    del buf172
    buf175 = buf171; del buf171  # reuse
    cpp_fused_convolution_backward_20(c_void_p(buf169.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf175.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf176 = aten.convolution_backward(buf175, constant_pad_nd_13, primals_106, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
    del constant_pad_nd_13
    del primals_106
    buf177 = buf176[0]
    buf178 = buf176[1]
    del buf176
    buf179 = buf175; del buf175  # reuse
    cpp_fused_convolution_backward_21(c_void_p(buf169.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf179.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf180 = aten.convolution_backward(buf179, constant_pad_nd_12, primals_105, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
    del constant_pad_nd_12
    del primals_105
    buf181 = buf180[0]
    buf182 = buf180[1]
    del buf180
    buf183 = buf179; del buf179  # reuse
    cpp_fused_convolution_backward_22(c_void_p(buf169.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf183.data_ptr()))
    del buf169
    del primals_108
    del squeeze_139
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf184 = aten.convolution_backward(buf183, constant_pad_nd_11, primals_104, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf183
    del constant_pad_nd_11
    del primals_104
    buf185 = buf184[0]
    buf186 = buf184[1]
    del buf184
    buf187 = empty_strided((8, 960, 14, 14), (188160, 1, 13440, 960), device='cpu', dtype=torch.float32)
    buf188 = empty((960, ), device='cpu', dtype=torch.float32)
    buf189 = empty((960, ), device='cpu', dtype=torch.float32)
    buf190 = empty((960, ), device='cpu', dtype=torch.float32)
    buf191 = buf187; del buf187  # reuse
    cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_23(c_void_p(buf191.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(mul_628.data_ptr()), c_void_p(convolution_119.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    del buf173
    del buf177
    del buf181
    del buf185
    del buf189
    del convolution_119
    del mul_628
    del primals_102
    del squeeze_136
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf192 = aten.convolution_backward(buf191, add_235, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_235
    del buf191
    del primals_264
    buf193 = buf192[0]
    buf194 = buf192[1]
    del buf192
    buf195 = empty((160, ), device='cpu', dtype=torch.float32)
    buf196 = empty((160, ), device='cpu', dtype=torch.float32)
    buf197 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    buf198 = buf196; del buf196  # reuse
    cpp_fused_native_batch_norm_backward_24(c_void_p(buf198.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(cat_33.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()))
    del cat_33
    del primals_100
    del squeeze_133
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf199 = aten.convolution_backward(reinterpret_tensor(buf197, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), getitem_321, primals_263, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_321
    del primals_263
    buf200 = buf199[0]
    buf201 = buf199[1]
    del buf199
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf202 = aten.convolution_backward(reinterpret_tensor(buf197, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), getitem_320, primals_262, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_320
    del primals_262
    buf203 = buf202[0]
    buf204 = buf202[1]
    del buf202
    buf205 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf206 = reinterpret_tensor(buf205, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf205  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_25(c_void_p(buf206.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(add_229.data_ptr()), c_void_p(convolution_116.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf207 = aten.convolution_backward(buf206, mul_354, primals_260, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf206
    del mul_354
    del primals_260
    buf208 = buf207[0]
    buf209 = buf207[1]
    buf210 = buf207[2]
    del buf207
    buf211 = reinterpret_tensor(buf208, (8, 80, 1, 1), (80, 1, 1, 1), 0); del buf208  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_26(c_void_p(buf211.data_ptr()), c_void_p(convolution_115.data_ptr()))
    del convolution_115
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf212 = aten.convolution_backward(buf211, mean_11, primals_258, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf211
    del mean_11
    del primals_258
    buf213 = buf212[0]
    buf214 = buf212[1]
    buf215 = buf212[2]
    del buf212
    buf216 = empty((8, 480, 14, 14), device='cpu', dtype=torch.float32)
    buf217 = empty((480, ), device='cpu', dtype=torch.float32)
    buf218 = empty((480, ), device='cpu', dtype=torch.float32)
    buf219 = buf216; del buf216  # reuse
    buf220 = buf218; del buf218  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_27(c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(convolution_116.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(add_229.data_ptr()), c_void_p(cat_32.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf217.data_ptr()))
    del add_229
    del buf200
    del buf203
    del cat_32
    del convolution_116
    del primals_98
    del squeeze_130
    del unsqueeze_402
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf221 = aten.convolution_backward(reinterpret_tensor(buf219, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), getitem_317, primals_257, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_317
    del primals_257
    buf222 = buf221[0]
    buf223 = buf221[1]
    del buf221
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf224 = aten.convolution_backward(reinterpret_tensor(buf219, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), getitem_312, primals_256, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_312
    del primals_256
    buf225 = buf224[0]
    buf226 = buf224[1]
    del buf224
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf227 = aten.convolution_backward(reinterpret_tensor(buf219, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), getitem_307, primals_255, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_307
    del primals_255
    buf228 = buf227[0]
    buf229 = buf227[1]
    del buf227
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf230 = aten.convolution_backward(reinterpret_tensor(buf219, (8, 120, 14, 14), (94080, 196, 14, 1), 0), getitem_302, primals_254, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_302
    del primals_254
    buf231 = buf230[0]
    buf232 = buf230[1]
    del buf230
    buf233 = buf219; del buf219  # reuse
    buf234 = empty((480, ), device='cpu', dtype=torch.float32)
    buf235 = empty((480, ), device='cpu', dtype=torch.float32)
    buf236 = buf233; del buf233  # reuse
    buf237 = buf235; del buf235  # reuse
    cpp_fused_cat_mul_native_batch_norm_backward_28(c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(mul_668.data_ptr()), c_void_p(cat_31.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf234.data_ptr()))
    del buf222
    del buf225
    del buf228
    del buf231
    del cat_31
    del mul_668
    del primals_96
    del squeeze_127
    del unsqueeze_414
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf238 = aten.convolution_backward(reinterpret_tensor(buf236, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), getitem_295, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_295
    del primals_253
    buf239 = buf238[0]
    buf240 = buf238[1]
    del buf238
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf241 = aten.convolution_backward(reinterpret_tensor(buf236, (8, 240, 14, 14), (94080, 196, 14, 1), 0), getitem_294, primals_252, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_294
    del primals_252
    buf242 = buf241[0]
    buf243 = buf241[1]
    del buf241
    buf244 = empty((160, ), device='cpu', dtype=torch.float32)
    buf245 = empty((160, ), device='cpu', dtype=torch.float32)
    buf246 = buf197; del buf197  # reuse
    buf247 = buf245; del buf245  # reuse
    cpp_fused_add_cat_native_batch_norm_backward_29(c_void_p(buf247.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(cat_30.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()))
    del cat_30
    del primals_94
    del squeeze_124
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf248 = aten.convolution_backward(reinterpret_tensor(buf246, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), getitem_291, primals_251, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_291
    del primals_251
    buf249 = buf248[0]
    buf250 = buf248[1]
    del buf248
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf251 = aten.convolution_backward(reinterpret_tensor(buf246, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), getitem_290, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_290
    del primals_250
    buf252 = buf251[0]
    buf253 = buf251[1]
    del buf251
    buf254 = reinterpret_tensor(buf213, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf213  # reuse
    buf255 = reinterpret_tensor(buf254, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf254  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_30(c_void_p(buf255.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(add_213.data_ptr()), c_void_p(convolution_106.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf256 = aten.convolution_backward(buf255, mul_329, primals_248, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf255
    del mul_329
    del primals_248
    buf257 = buf256[0]
    buf258 = buf256[1]
    buf259 = buf256[2]
    del buf256
    buf260 = reinterpret_tensor(buf257, (8, 80, 1, 1), (80, 1, 1, 1), 0); del buf257  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_31(c_void_p(buf260.data_ptr()), c_void_p(convolution_105.data_ptr()))
    del convolution_105
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf261 = aten.convolution_backward(buf260, mean_10, primals_246, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf260
    del mean_10
    del primals_246
    buf262 = buf261[0]
    buf263 = buf261[1]
    buf264 = buf261[2]
    del buf261
    buf265 = buf236; del buf236  # reuse
    buf266 = empty((480, ), device='cpu', dtype=torch.float32)
    buf267 = empty((480, ), device='cpu', dtype=torch.float32)
    buf268 = buf265; del buf265  # reuse
    buf269 = buf267; del buf267  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_32(c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(convolution_106.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(add_213.data_ptr()), c_void_p(cat_29.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf266.data_ptr()))
    del add_213
    del buf249
    del buf252
    del cat_29
    del convolution_106
    del primals_92
    del squeeze_121
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf270 = aten.convolution_backward(reinterpret_tensor(buf268, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), getitem_287, primals_245, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_287
    del primals_245
    buf271 = buf270[0]
    buf272 = buf270[1]
    del buf270
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf273 = aten.convolution_backward(reinterpret_tensor(buf268, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), getitem_282, primals_244, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_282
    del primals_244
    buf274 = buf273[0]
    buf275 = buf273[1]
    del buf273
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf276 = aten.convolution_backward(reinterpret_tensor(buf268, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), getitem_277, primals_243, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_277
    del primals_243
    buf277 = buf276[0]
    buf278 = buf276[1]
    del buf276
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf279 = aten.convolution_backward(reinterpret_tensor(buf268, (8, 120, 14, 14), (94080, 196, 14, 1), 0), getitem_272, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_272
    del primals_242
    buf280 = buf279[0]
    buf281 = buf279[1]
    del buf279
    buf282 = buf268; del buf268  # reuse
    buf283 = empty((480, ), device='cpu', dtype=torch.float32)
    buf284 = empty((480, ), device='cpu', dtype=torch.float32)
    buf285 = buf282; del buf282  # reuse
    buf286 = buf284; del buf284  # reuse
    cpp_fused_cat_mul_native_batch_norm_backward_33(c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(mul_708.data_ptr()), c_void_p(cat_28.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf283.data_ptr()))
    del buf271
    del buf274
    del buf277
    del buf280
    del cat_28
    del mul_708
    del primals_90
    del squeeze_118
    del unsqueeze_450
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf287 = aten.convolution_backward(reinterpret_tensor(buf285, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), getitem_265, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_265
    del primals_241
    buf288 = buf287[0]
    buf289 = buf287[1]
    del buf287
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf290 = aten.convolution_backward(reinterpret_tensor(buf285, (8, 240, 14, 14), (94080, 196, 14, 1), 0), getitem_264, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_264
    del primals_240
    buf291 = buf290[0]
    buf292 = buf290[1]
    del buf290
    buf293 = buf193; del buf193  # reuse
    buf294 = empty((160, ), device='cpu', dtype=torch.float32)
    buf295 = empty((160, ), device='cpu', dtype=torch.float32)
    buf296 = reinterpret_tensor(buf246, (8, 160, 14, 14), (31360, 196, 14, 1), 0); del buf246  # reuse
    buf297 = buf295; del buf295  # reuse
    buf298 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_convolution_backward_native_batch_norm_backward_34(c_void_p(buf293.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(cat_27.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()))
    del buf239
    del buf242
    del buf288
    del buf291
    del cat_27
    del primals_88
    del squeeze_115
    del unsqueeze_462
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf299 = aten.convolution_backward(buf298, getitem_261, primals_239, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_261
    del primals_239
    buf300 = buf299[0]
    buf301 = buf299[1]
    del buf299
    buf302 = buf298; del buf298  # reuse
    cpp_fused_convolution_backward_35(c_void_p(buf296.data_ptr()), c_void_p(buf302.data_ptr()))
    del buf296
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf303 = aten.convolution_backward(buf302, getitem_260, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf302
    del getitem_260
    del primals_238
    buf304 = buf303[0]
    buf305 = buf303[1]
    del buf303
    buf306 = reinterpret_tensor(buf262, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf262  # reuse
    buf307 = reinterpret_tensor(buf306, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf306  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_36(c_void_p(buf307.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(add_197.data_ptr()), c_void_p(convolution_96.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf308 = aten.convolution_backward(buf307, mul_304, primals_236, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf307
    del mul_304
    del primals_236
    buf309 = buf308[0]
    buf310 = buf308[1]
    buf311 = buf308[2]
    del buf308
    buf312 = reinterpret_tensor(buf309, (8, 80, 1, 1), (80, 1, 1, 1), 0); del buf309  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_37(c_void_p(buf312.data_ptr()), c_void_p(convolution_95.data_ptr()))
    del convolution_95
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf313 = aten.convolution_backward(buf312, mean_9, primals_234, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf312
    del mean_9
    del primals_234
    buf314 = buf313[0]
    buf315 = buf313[1]
    buf316 = buf313[2]
    del buf313
    buf317 = buf285; del buf285  # reuse
    buf318 = empty((480, ), device='cpu', dtype=torch.float32)
    buf319 = empty((480, ), device='cpu', dtype=torch.float32)
    buf320 = buf317; del buf317  # reuse
    buf321 = buf319; del buf319  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_38(c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(convolution_96.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(add_197.data_ptr()), c_void_p(cat_26.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf318.data_ptr()))
    del add_197
    del buf300
    del buf314
    del cat_26
    del convolution_96
    del primals_86
    del squeeze_112
    del unsqueeze_474
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf322 = aten.convolution_backward(reinterpret_tensor(buf320, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), getitem_257, primals_233, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_257
    del primals_233
    buf323 = buf322[0]
    buf324 = buf322[1]
    del buf322
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf325 = aten.convolution_backward(reinterpret_tensor(buf320, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), getitem_252, primals_232, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_252
    del primals_232
    buf326 = buf325[0]
    buf327 = buf325[1]
    del buf325
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf328 = aten.convolution_backward(reinterpret_tensor(buf320, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), getitem_247, primals_231, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_247
    del primals_231
    buf329 = buf328[0]
    buf330 = buf328[1]
    del buf328
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf331 = aten.convolution_backward(reinterpret_tensor(buf320, (8, 120, 14, 14), (94080, 196, 14, 1), 0), getitem_242, primals_230, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
    del getitem_242
    del primals_230
    buf332 = buf331[0]
    buf333 = buf331[1]
    del buf331
    buf334 = buf320; del buf320  # reuse
    buf335 = empty((480, ), device='cpu', dtype=torch.float32)
    buf336 = empty((480, ), device='cpu', dtype=torch.float32)
    buf337 = buf334; del buf334  # reuse
    buf338 = buf336; del buf336  # reuse
    cpp_fused_cat_mul_native_batch_norm_backward_39(c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(mul_748.data_ptr()), c_void_p(cat_25.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf335.data_ptr()))
    del buf323
    del buf326
    del buf329
    del buf332
    del cat_25
    del mul_748
    del primals_84
    del squeeze_109
    del unsqueeze_486
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf339 = aten.convolution_backward(reinterpret_tensor(buf337, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), getitem_235, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_235
    del primals_229
    buf340 = buf339[0]
    buf341 = buf339[1]
    del buf339
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf342 = aten.convolution_backward(reinterpret_tensor(buf337, (8, 240, 14, 14), (94080, 196, 14, 1), 0), getitem_234, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf337
    del getitem_234
    del primals_228
    buf343 = buf342[0]
    buf344 = buf342[1]
    del buf342
    buf345 = empty((160, ), device='cpu', dtype=torch.float32)
    buf346 = empty((160, ), device='cpu', dtype=torch.float32)
    buf347 = buf293; del buf293  # reuse
    buf348 = buf346; del buf346  # reuse
    cpp_fused_add_cat_native_batch_norm_backward_40(c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(convolution_88.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf345.data_ptr()))
    del buf340
    del buf343
    del convolution_88
    del primals_82
    del squeeze_106
    del unsqueeze_498
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf349 = aten.convolution_backward(buf347, mul_280, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf347
    del mul_280
    del primals_227
    buf350 = buf349[0]
    buf351 = buf349[1]
    del buf349
    buf352 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cpu', dtype=torch.float32)
    buf353 = reinterpret_tensor(buf352, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf352  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_41(c_void_p(buf353.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(add_182.data_ptr()), c_void_p(convolution_87.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf354 = aten.convolution_backward(buf353, mul_279, primals_225, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf353
    del mul_279
    del primals_225
    buf355 = buf354[0]
    buf356 = buf354[1]
    buf357 = buf354[2]
    del buf354
    buf358 = reinterpret_tensor(buf355, (8, 52, 1, 1), (52, 1, 1, 1), 0); del buf355  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_42(c_void_p(buf358.data_ptr()), c_void_p(convolution_86.data_ptr()))
    del convolution_86
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf359 = aten.convolution_backward(buf358, mean_8, primals_223, [52], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf358
    del mean_8
    del primals_223
    buf360 = buf359[0]
    buf361 = buf359[1]
    buf362 = buf359[2]
    del buf359
    buf363 = empty((624, ), device='cpu', dtype=torch.float32)
    buf364 = empty((624, ), device='cpu', dtype=torch.float32)
    buf365 = buf350; del buf350  # reuse
    buf366 = buf364; del buf364  # reuse
    buf367 = buf365; del buf365  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_43(c_void_p(buf367.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(convolution_87.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(add_182.data_ptr()), c_void_p(convolution_85.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf363.data_ptr()))
    del add_182
    del convolution_85
    del convolution_87
    del primals_80
    del squeeze_103
    del unsqueeze_510
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf368 = aten.convolution_backward(buf367, mul_270, primals_222, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 624, [True, True, False])
    del buf367
    del mul_270
    del primals_222
    buf369 = buf368[0]
    buf370 = buf368[1]
    del buf368
    buf371 = empty((624, ), device='cpu', dtype=torch.float32)
    buf372 = empty((624, ), device='cpu', dtype=torch.float32)
    buf373 = empty((624, ), device='cpu', dtype=torch.float32)
    buf374 = buf369; del buf369  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_44(c_void_p(buf374.data_ptr()), c_void_p(mul_788.data_ptr()), c_void_p(convolution_84.data_ptr()), c_void_p(unsqueeze_522.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()))
    del convolution_84
    del mul_788
    del primals_78
    del squeeze_100
    del unsqueeze_522
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf375 = aten.convolution_backward(buf374, add_172, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_172
    del primals_221
    buf376 = buf375[0]
    buf377 = buf375[1]
    del buf375
    buf378 = empty((104, ), device='cpu', dtype=torch.float32)
    buf379 = empty((104, ), device='cpu', dtype=torch.float32)
    buf380 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    buf381 = buf379; del buf379  # reuse
    cpp_fused_native_batch_norm_backward_45(c_void_p(buf381.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(cat_24.data_ptr()), c_void_p(unsqueeze_534.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf380.data_ptr()))
    del cat_24
    del primals_76
    del squeeze_97
    del unsqueeze_534
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf382 = aten.convolution_backward(reinterpret_tensor(buf380, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), getitem_225, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_225
    del primals_220
    buf383 = buf382[0]
    buf384 = buf382[1]
    del buf382
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf385 = aten.convolution_backward(reinterpret_tensor(buf380, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), getitem_224, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_224
    del primals_219
    buf386 = buf385[0]
    buf387 = buf385[1]
    del buf385
    buf388 = reinterpret_tensor(buf360, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf360  # reuse
    buf389 = reinterpret_tensor(buf388, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf388  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_46(c_void_p(buf389.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(add_166.data_ptr()), c_void_p(convolution_81.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf390 = aten.convolution_backward(buf389, mul_254, primals_217, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf389
    del mul_254
    del primals_217
    buf391 = buf390[0]
    buf392 = buf390[1]
    buf393 = buf390[2]
    del buf390
    buf394 = reinterpret_tensor(buf391, (8, 26, 1, 1), (26, 1, 1, 1), 0); del buf391  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_47(c_void_p(buf394.data_ptr()), c_void_p(convolution_80.data_ptr()))
    del convolution_80
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf395 = aten.convolution_backward(buf394, mean_7, primals_215, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf394
    del mean_7
    del primals_215
    buf396 = buf395[0]
    buf397 = buf395[1]
    buf398 = buf395[2]
    del buf395
    buf399 = reinterpret_tensor(buf374, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf374  # reuse
    buf400 = buf372; del buf372  # reuse
    buf401 = empty((624, ), device='cpu', dtype=torch.float32)
    buf402 = buf399; del buf399  # reuse
    buf403 = buf401; del buf401  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_48(c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(convolution_81.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(add_166.data_ptr()), c_void_p(cat_23.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf400.data_ptr()))
    del add_166
    del buf383
    del buf386
    del cat_23
    del convolution_81
    del primals_74
    del squeeze_94
    del unsqueeze_546
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf404 = aten.convolution_backward(reinterpret_tensor(buf402, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), getitem_221, primals_214, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_221
    del primals_214
    buf405 = buf404[0]
    buf406 = buf404[1]
    del buf404
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf407 = aten.convolution_backward(reinterpret_tensor(buf402, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), getitem_216, primals_213, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_216
    del primals_213
    buf408 = buf407[0]
    buf409 = buf407[1]
    del buf407
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf410 = aten.convolution_backward(reinterpret_tensor(buf402, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), getitem_211, primals_212, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_211
    del primals_212
    buf411 = buf410[0]
    buf412 = buf410[1]
    del buf410
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf413 = aten.convolution_backward(reinterpret_tensor(buf402, (8, 156, 14, 14), (122304, 196, 14, 1), 0), getitem_206, primals_211, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_206
    del primals_211
    buf414 = buf413[0]
    buf415 = buf413[1]
    del buf413
    buf416 = buf402; del buf402  # reuse
    buf417 = empty((624, ), device='cpu', dtype=torch.float32)
    buf418 = empty((624, ), device='cpu', dtype=torch.float32)
    buf419 = buf416; del buf416  # reuse
    buf420 = buf418; del buf418  # reuse
    cpp_fused_cat_mul_native_batch_norm_backward_49(c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(mul_828.data_ptr()), c_void_p(cat_22.data_ptr()), c_void_p(unsqueeze_558.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf417.data_ptr()))
    del buf405
    del buf408
    del buf411
    del buf414
    del cat_22
    del mul_828
    del primals_72
    del squeeze_91
    del unsqueeze_558
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf421 = aten.convolution_backward(reinterpret_tensor(buf419, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), getitem_199, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_199
    del primals_210
    buf422 = buf421[0]
    buf423 = buf421[1]
    del buf421
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf424 = aten.convolution_backward(reinterpret_tensor(buf419, (8, 312, 14, 14), (122304, 196, 14, 1), 0), getitem_198, primals_209, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_198
    del primals_209
    buf425 = buf424[0]
    buf426 = buf424[1]
    del buf424
    buf427 = empty((104, ), device='cpu', dtype=torch.float32)
    buf428 = empty((104, ), device='cpu', dtype=torch.float32)
    buf429 = buf380; del buf380  # reuse
    buf430 = buf428; del buf428  # reuse
    cpp_fused_add_cat_native_batch_norm_backward_50(c_void_p(buf430.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(cat_21.data_ptr()), c_void_p(unsqueeze_570.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf429.data_ptr()))
    del cat_21
    del primals_70
    del squeeze_88
    del unsqueeze_570
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf431 = aten.convolution_backward(reinterpret_tensor(buf429, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), getitem_195, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_195
    del primals_208
    buf432 = buf431[0]
    buf433 = buf431[1]
    del buf431
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf434 = aten.convolution_backward(reinterpret_tensor(buf429, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), getitem_194, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_194
    del primals_207
    buf435 = buf434[0]
    buf436 = buf434[1]
    del buf434
    buf437 = reinterpret_tensor(buf396, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf396  # reuse
    buf438 = reinterpret_tensor(buf437, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf437  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_51(c_void_p(buf438.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(add_150.data_ptr()), c_void_p(convolution_71.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf439 = aten.convolution_backward(buf438, mul_229, primals_205, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf438
    del mul_229
    del primals_205
    buf440 = buf439[0]
    buf441 = buf439[1]
    buf442 = buf439[2]
    del buf439
    buf443 = reinterpret_tensor(buf440, (8, 26, 1, 1), (26, 1, 1, 1), 0); del buf440  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_52(c_void_p(buf443.data_ptr()), c_void_p(convolution_70.data_ptr()))
    del convolution_70
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf444 = aten.convolution_backward(buf443, mean_6, primals_203, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf443
    del mean_6
    del primals_203
    buf445 = buf444[0]
    buf446 = buf444[1]
    buf447 = buf444[2]
    del buf444
    buf448 = buf419; del buf419  # reuse
    buf449 = empty((624, ), device='cpu', dtype=torch.float32)
    buf450 = empty((624, ), device='cpu', dtype=torch.float32)
    buf451 = buf448; del buf448  # reuse
    buf452 = buf450; del buf450  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_53(c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(add_150.data_ptr()), c_void_p(cat_20.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf449.data_ptr()))
    del add_150
    del buf432
    del buf435
    del cat_20
    del convolution_71
    del primals_68
    del squeeze_85
    del unsqueeze_582
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf453 = aten.convolution_backward(reinterpret_tensor(buf451, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), getitem_191, primals_202, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_191
    del primals_202
    buf454 = buf453[0]
    buf455 = buf453[1]
    del buf453
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf456 = aten.convolution_backward(reinterpret_tensor(buf451, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), getitem_186, primals_201, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_186
    del primals_201
    buf457 = buf456[0]
    buf458 = buf456[1]
    del buf456
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf459 = aten.convolution_backward(reinterpret_tensor(buf451, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), getitem_181, primals_200, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_181
    del primals_200
    buf460 = buf459[0]
    buf461 = buf459[1]
    del buf459
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf462 = aten.convolution_backward(reinterpret_tensor(buf451, (8, 156, 14, 14), (122304, 196, 14, 1), 0), getitem_176, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_176
    del primals_199
    buf463 = buf462[0]
    buf464 = buf462[1]
    del buf462
    buf465 = buf451; del buf451  # reuse
    buf466 = empty((624, ), device='cpu', dtype=torch.float32)
    buf467 = empty((624, ), device='cpu', dtype=torch.float32)
    buf468 = buf465; del buf465  # reuse
    buf469 = buf467; del buf467  # reuse
    cpp_fused_cat_mul_native_batch_norm_backward_54(c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(mul_868.data_ptr()), c_void_p(cat_19.data_ptr()), c_void_p(unsqueeze_594.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf466.data_ptr()))
    del buf454
    del buf457
    del buf460
    del buf463
    del cat_19
    del mul_868
    del primals_66
    del squeeze_82
    del unsqueeze_594
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf470 = aten.convolution_backward(reinterpret_tensor(buf468, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), getitem_169, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_169
    del primals_198
    buf471 = buf470[0]
    buf472 = buf470[1]
    del buf470
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf473 = aten.convolution_backward(reinterpret_tensor(buf468, (8, 312, 14, 14), (122304, 196, 14, 1), 0), getitem_168, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_168
    del primals_197
    buf474 = buf473[0]
    buf475 = buf473[1]
    del buf473
    buf476 = buf376; del buf376  # reuse
    buf477 = empty((104, ), device='cpu', dtype=torch.float32)
    buf478 = empty((104, ), device='cpu', dtype=torch.float32)
    buf479 = reinterpret_tensor(buf429, (8, 104, 14, 14), (20384, 196, 14, 1), 0); del buf429  # reuse
    buf480 = buf478; del buf478  # reuse
    buf481 = empty_strided((8, 52, 14, 14), (10192, 1, 728, 52), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_convolution_backward_native_batch_norm_backward_55(c_void_p(buf476.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(cat_18.data_ptr()), c_void_p(unsqueeze_606.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf481.data_ptr()))
    del buf422
    del buf425
    del buf471
    del buf474
    del cat_18
    del primals_64
    del squeeze_79
    del unsqueeze_606
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf482 = aten.convolution_backward(buf481, getitem_165, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_165
    del primals_196
    buf483 = buf482[0]
    buf484 = buf482[1]
    del buf482
    buf485 = buf481; del buf481  # reuse
    cpp_fused_convolution_backward_56(c_void_p(buf479.data_ptr()), c_void_p(buf485.data_ptr()))
    del buf479
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf486 = aten.convolution_backward(buf485, getitem_164, primals_195, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf485
    del getitem_164
    del primals_195
    buf487 = buf486[0]
    buf488 = buf486[1]
    del buf486
    buf489 = reinterpret_tensor(buf445, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf445  # reuse
    buf490 = reinterpret_tensor(buf489, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf489  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_57(c_void_p(buf490.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(add_134.data_ptr()), c_void_p(convolution_61.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf491 = aten.convolution_backward(buf490, mul_204, primals_193, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf490
    del mul_204
    del primals_193
    buf492 = buf491[0]
    buf493 = buf491[1]
    buf494 = buf491[2]
    del buf491
    buf495 = reinterpret_tensor(buf492, (8, 26, 1, 1), (26, 1, 1, 1), 0); del buf492  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_58(c_void_p(buf495.data_ptr()), c_void_p(convolution_60.data_ptr()))
    del convolution_60
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf496 = aten.convolution_backward(buf495, mean_5, primals_191, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf495
    del mean_5
    del primals_191
    buf497 = buf496[0]
    buf498 = buf496[1]
    buf499 = buf496[2]
    del buf496
    buf500 = buf468; del buf468  # reuse
    buf501 = empty((624, ), device='cpu', dtype=torch.float32)
    buf502 = empty((624, ), device='cpu', dtype=torch.float32)
    buf503 = buf500; del buf500  # reuse
    buf504 = buf502; del buf502  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_59(c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(add_134.data_ptr()), c_void_p(cat_17.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf501.data_ptr()))
    del add_134
    del buf483
    del buf487
    del buf497
    del cat_17
    del convolution_61
    del primals_62
    del squeeze_76
    del unsqueeze_618
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf505 = aten.convolution_backward(reinterpret_tensor(buf503, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), getitem_161, primals_190, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_161
    del primals_190
    buf506 = buf505[0]
    buf507 = buf505[1]
    del buf505
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf508 = aten.convolution_backward(reinterpret_tensor(buf503, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), getitem_156, primals_189, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_156
    del primals_189
    buf509 = buf508[0]
    buf510 = buf508[1]
    del buf508
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf511 = aten.convolution_backward(reinterpret_tensor(buf503, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), getitem_151, primals_188, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_151
    del primals_188
    buf512 = buf511[0]
    buf513 = buf511[1]
    del buf511
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf514 = aten.convolution_backward(reinterpret_tensor(buf503, (8, 156, 14, 14), (122304, 196, 14, 1), 0), getitem_146, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False])
    del getitem_146
    del primals_187
    buf515 = buf514[0]
    buf516 = buf514[1]
    del buf514
    buf517 = buf503; del buf503  # reuse
    buf518 = empty((624, ), device='cpu', dtype=torch.float32)
    buf519 = empty((624, ), device='cpu', dtype=torch.float32)
    buf520 = buf517; del buf517  # reuse
    buf521 = buf519; del buf519  # reuse
    cpp_fused_cat_mul_native_batch_norm_backward_60(c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(mul_908.data_ptr()), c_void_p(cat_16.data_ptr()), c_void_p(unsqueeze_630.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf518.data_ptr()))
    del buf506
    del buf509
    del buf512
    del buf515
    del cat_16
    del mul_908
    del primals_60
    del squeeze_73
    del unsqueeze_630
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf522 = aten.convolution_backward(reinterpret_tensor(buf520, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), getitem_139, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_139
    del primals_186
    buf523 = buf522[0]
    buf524 = buf522[1]
    del buf522
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf525 = aten.convolution_backward(reinterpret_tensor(buf520, (8, 312, 14, 14), (122304, 196, 14, 1), 0), getitem_138, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf520
    del getitem_138
    del primals_185
    buf526 = buf525[0]
    buf527 = buf525[1]
    del buf525
    buf528 = empty((104, ), device='cpu', dtype=torch.float32)
    buf529 = empty((104, ), device='cpu', dtype=torch.float32)
    buf530 = buf476; del buf476  # reuse
    buf531 = buf529; del buf529  # reuse
    cpp_fused_add_cat_native_batch_norm_backward_61(c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_642.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf528.data_ptr()))
    del buf523
    del buf526
    del convolution_53
    del primals_58
    del squeeze_70
    del unsqueeze_642
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf532 = aten.convolution_backward(buf530, mul_180, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf530
    del mul_180
    del primals_184
    buf533 = buf532[0]
    buf534 = buf532[1]
    del buf532
    buf535 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cpu', dtype=torch.float32)
    buf536 = reinterpret_tensor(buf535, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf535  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_62(c_void_p(buf536.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(add_119.data_ptr()), c_void_p(convolution_52.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf537 = aten.convolution_backward(buf536, mul_179, primals_182, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf536
    del mul_179
    del primals_182
    buf538 = buf537[0]
    buf539 = buf537[1]
    buf540 = buf537[2]
    del buf537
    buf541 = reinterpret_tensor(buf538, (8, 14, 1, 1), (14, 1, 1, 1), 0); del buf538  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_63(c_void_p(buf541.data_ptr()), c_void_p(convolution_51.data_ptr()))
    del convolution_51
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf542 = aten.convolution_backward(buf541, mean_4, primals_180, [14], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf541
    del mean_4
    del primals_180
    buf543 = buf542[0]
    buf544 = buf542[1]
    buf545 = buf542[2]
    del buf542
    buf546 = empty((336, ), device='cpu', dtype=torch.float32)
    buf547 = empty((336, ), device='cpu', dtype=torch.float32)
    buf548 = buf533; del buf533  # reuse
    buf549 = buf547; del buf547  # reuse
    buf550 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64(c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(add_119.data_ptr()), c_void_p(cat_15.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf550.data_ptr()))
    del add_119
    del cat_15
    del convolution_52
    del unsqueeze_654
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf551 = aten.convolution_backward(buf550, constant_pad_nd_10, primals_55, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 112, [True, True, False])
    del constant_pad_nd_10
    del primals_55
    buf552 = buf551[0]
    buf553 = buf551[1]
    del buf551
    buf554 = buf550; del buf550  # reuse
    cpp_fused_convolution_backward_65(c_void_p(buf548.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf554.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf555 = aten.convolution_backward(buf554, constant_pad_nd_9, primals_54, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 112, [True, True, False])
    del constant_pad_nd_9
    del primals_54
    buf556 = buf555[0]
    buf557 = buf555[1]
    del buf555
    buf558 = buf554; del buf554  # reuse
    cpp_fused_convolution_backward_66(c_void_p(buf548.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf558.data_ptr()))
    del buf548
    del primals_56
    del squeeze_67
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf559 = aten.convolution_backward(buf558, constant_pad_nd_8, primals_53, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 112, [True, True, False])
    del constant_pad_nd_8
    del primals_53
    buf560 = buf559[0]
    buf561 = buf559[1]
    del buf559
    buf562 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    buf563 = empty((336, ), device='cpu', dtype=torch.float32)
    buf564 = empty((336, ), device='cpu', dtype=torch.float32)
    buf565 = empty((336, ), device='cpu', dtype=torch.float32)
    buf566 = buf562; del buf562  # reuse
    cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_67(c_void_p(buf566.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(mul_948.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_666.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()))
    del buf552
    del buf556
    del buf560
    del convolution_47
    del mul_948
    del primals_51
    del squeeze_64
    del unsqueeze_666
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf567 = aten.convolution_backward(buf566, add_109, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_109
    del primals_179
    buf568 = buf567[0]
    buf569 = buf567[1]
    del buf567
    buf570 = empty((56, ), device='cpu', dtype=torch.float32)
    buf571 = empty((56, ), device='cpu', dtype=torch.float32)
    buf572 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    buf573 = buf571; del buf571  # reuse
    cpp_fused_native_batch_norm_backward_68(c_void_p(buf573.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(cat_14.data_ptr()), c_void_p(unsqueeze_678.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf572.data_ptr()))
    del cat_14
    del primals_49
    del squeeze_61
    del unsqueeze_678
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf574 = aten.convolution_backward(reinterpret_tensor(buf572, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), getitem_117, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_117
    del primals_178
    buf575 = buf574[0]
    buf576 = buf574[1]
    del buf574
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf577 = aten.convolution_backward(reinterpret_tensor(buf572, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), getitem_116, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_116
    del primals_177
    buf578 = buf577[0]
    buf579 = buf577[1]
    del buf577
    buf580 = reinterpret_tensor(buf543, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf543  # reuse
    buf581 = reinterpret_tensor(buf580, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf580  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_69(c_void_p(buf581.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(add_103.data_ptr()), c_void_p(convolution_44.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf582 = aten.convolution_backward(buf581, mul_154, primals_175, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf581
    del mul_154
    del primals_175
    buf583 = buf582[0]
    buf584 = buf582[1]
    buf585 = buf582[2]
    del buf582
    buf586 = reinterpret_tensor(buf583, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf583  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_70(c_void_p(buf586.data_ptr()), c_void_p(convolution_43.data_ptr()))
    del convolution_43
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf587 = aten.convolution_backward(buf586, mean_3, primals_173, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf586
    del mean_3
    del primals_173
    buf588 = buf587[0]
    buf589 = buf587[1]
    buf590 = buf587[2]
    del buf587
    buf591 = reinterpret_tensor(buf566, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf566  # reuse
    buf592 = buf564; del buf564  # reuse
    buf593 = empty((336, ), device='cpu', dtype=torch.float32)
    buf594 = buf591; del buf591  # reuse
    buf595 = buf593; del buf593  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71(c_void_p(buf594.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(add_103.data_ptr()), c_void_p(cat_13.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf592.data_ptr()))
    del add_103
    del buf575
    del buf578
    del cat_13
    del convolution_44
    del primals_47
    del squeeze_58
    del unsqueeze_690
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf596 = aten.convolution_backward(reinterpret_tensor(buf594, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_113, primals_172, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False])
    del getitem_113
    del primals_172
    buf597 = buf596[0]
    buf598 = buf596[1]
    del buf596
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf599 = aten.convolution_backward(reinterpret_tensor(buf594, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_110, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False])
    del getitem_110
    del primals_171
    buf600 = buf599[0]
    buf601 = buf599[1]
    del buf599
    buf602 = empty((336, ), device='cpu', dtype=torch.float32)
    buf603 = empty((336, ), device='cpu', dtype=torch.float32)
    buf604 = buf594; del buf594  # reuse
    buf605 = buf603; del buf603  # reuse
    cpp_fused_cat_mul_native_batch_norm_backward_72(c_void_p(buf605.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(mul_988.data_ptr()), c_void_p(cat_12.data_ptr()), c_void_p(unsqueeze_702.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf604.data_ptr()))
    del buf597
    del buf600
    del cat_12
    del mul_988
    del primals_45
    del squeeze_55
    del unsqueeze_702
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf606 = aten.convolution_backward(reinterpret_tensor(buf604, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_105, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_105
    del primals_170
    buf607 = buf606[0]
    buf608 = buf606[1]
    del buf606
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf609 = aten.convolution_backward(reinterpret_tensor(buf604, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_104, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_104
    del primals_169
    buf610 = buf609[0]
    buf611 = buf609[1]
    del buf609
    buf612 = empty((56, ), device='cpu', dtype=torch.float32)
    buf613 = empty((56, ), device='cpu', dtype=torch.float32)
    buf614 = buf572; del buf572  # reuse
    buf615 = buf613; del buf613  # reuse
    cpp_fused_add_cat_native_batch_norm_backward_73(c_void_p(buf615.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(cat_11.data_ptr()), c_void_p(unsqueeze_714.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf614.data_ptr()))
    del cat_11
    del primals_43
    del squeeze_52
    del unsqueeze_714
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf616 = aten.convolution_backward(reinterpret_tensor(buf614, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), getitem_101, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_101
    del primals_168
    buf617 = buf616[0]
    buf618 = buf616[1]
    del buf616
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf619 = aten.convolution_backward(reinterpret_tensor(buf614, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), getitem_100, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_100
    del primals_167
    buf620 = buf619[0]
    buf621 = buf619[1]
    del buf619
    buf622 = reinterpret_tensor(buf588, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf588  # reuse
    buf623 = reinterpret_tensor(buf622, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf622  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_74(c_void_p(buf623.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(add_87.data_ptr()), c_void_p(convolution_36.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf624 = aten.convolution_backward(buf623, mul_129, primals_165, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf623
    del mul_129
    del primals_165
    buf625 = buf624[0]
    buf626 = buf624[1]
    buf627 = buf624[2]
    del buf624
    buf628 = reinterpret_tensor(buf625, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf625  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_75(c_void_p(buf628.data_ptr()), c_void_p(convolution_35.data_ptr()))
    del convolution_35
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf629 = aten.convolution_backward(buf628, mean_2, primals_163, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf628
    del mean_2
    del primals_163
    buf630 = buf629[0]
    buf631 = buf629[1]
    buf632 = buf629[2]
    del buf629
    buf633 = buf604; del buf604  # reuse
    buf634 = empty((336, ), device='cpu', dtype=torch.float32)
    buf635 = empty((336, ), device='cpu', dtype=torch.float32)
    buf636 = buf633; del buf633  # reuse
    buf637 = buf635; del buf635  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_76(c_void_p(buf636.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(add_87.data_ptr()), c_void_p(cat_10.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf634.data_ptr()))
    del add_87
    del buf617
    del buf620
    del cat_10
    del convolution_36
    del primals_41
    del squeeze_49
    del unsqueeze_726
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf638 = aten.convolution_backward(reinterpret_tensor(buf636, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_97, primals_162, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False])
    del getitem_97
    del primals_162
    buf639 = buf638[0]
    buf640 = buf638[1]
    del buf638
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf641 = aten.convolution_backward(reinterpret_tensor(buf636, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_94, primals_161, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False])
    del getitem_94
    del primals_161
    buf642 = buf641[0]
    buf643 = buf641[1]
    del buf641
    buf644 = empty((336, ), device='cpu', dtype=torch.float32)
    buf645 = empty((336, ), device='cpu', dtype=torch.float32)
    buf646 = buf636; del buf636  # reuse
    buf647 = buf645; del buf645  # reuse
    cpp_fused_cat_mul_native_batch_norm_backward_77(c_void_p(buf647.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(mul_1028.data_ptr()), c_void_p(cat_9.data_ptr()), c_void_p(unsqueeze_738.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf646.data_ptr()))
    del buf639
    del buf642
    del cat_9
    del mul_1028
    del primals_39
    del squeeze_46
    del unsqueeze_738
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf648 = aten.convolution_backward(reinterpret_tensor(buf646, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_89, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_89
    del primals_160
    buf649 = buf648[0]
    buf650 = buf648[1]
    del buf648
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf651 = aten.convolution_backward(reinterpret_tensor(buf646, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_88, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_88
    del primals_159
    buf652 = buf651[0]
    buf653 = buf651[1]
    del buf651
    buf654 = buf568; del buf568  # reuse
    buf655 = empty((56, ), device='cpu', dtype=torch.float32)
    buf656 = empty((56, ), device='cpu', dtype=torch.float32)
    buf657 = reinterpret_tensor(buf614, (8, 56, 28, 28), (43904, 784, 28, 1), 0); del buf614  # reuse
    buf658 = buf656; del buf656  # reuse
    buf659 = reinterpret_tensor(buf558, (8, 28, 28, 28), (21952, 1, 784, 28), 0); del buf558  # reuse
    cpp_fused_add_cat_convolution_backward_native_batch_norm_backward_78(c_void_p(buf654.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(cat_8.data_ptr()), c_void_p(unsqueeze_750.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf659.data_ptr()))
    del buf607
    del buf610
    del buf649
    del buf652
    del cat_8
    del primals_37
    del squeeze_43
    del unsqueeze_750
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf660 = aten.convolution_backward(buf659, getitem_85, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_85
    del primals_158
    buf661 = buf660[0]
    buf662 = buf660[1]
    del buf660
    buf663 = buf659; del buf659  # reuse
    cpp_fused_convolution_backward_79(c_void_p(buf657.data_ptr()), c_void_p(buf663.data_ptr()))
    del buf657
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf664 = aten.convolution_backward(buf663, getitem_84, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf663
    del getitem_84
    del primals_157
    buf665 = buf664[0]
    buf666 = buf664[1]
    del buf664
    buf667 = reinterpret_tensor(buf630, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf630  # reuse
    buf668 = reinterpret_tensor(buf667, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf667  # reuse
    cpp_fused_cat_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_80(c_void_p(buf668.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(add_71.data_ptr()), c_void_p(convolution_28.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf669 = aten.convolution_backward(buf668, mul_104, primals_155, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf668
    del mul_104
    del primals_155
    buf670 = buf669[0]
    buf671 = buf669[1]
    buf672 = buf669[2]
    del buf669
    buf673 = reinterpret_tensor(buf670, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf670  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_81(c_void_p(buf673.data_ptr()), c_void_p(convolution_27.data_ptr()))
    del convolution_27
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf674 = aten.convolution_backward(buf673, mean_1, primals_153, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf673
    del mean_1
    del primals_153
    buf675 = buf674[0]
    buf676 = buf674[1]
    buf677 = buf674[2]
    del buf674
    buf678 = buf646; del buf646  # reuse
    buf679 = empty((336, ), device='cpu', dtype=torch.float32)
    buf680 = empty((336, ), device='cpu', dtype=torch.float32)
    buf681 = buf678; del buf678  # reuse
    buf682 = buf680; del buf680  # reuse
    cpp_fused_add_cat_div_fill_mul_native_batch_norm_backward_sigmoid_sub_82(c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(add_71.data_ptr()), c_void_p(cat_7.data_ptr()), c_void_p(unsqueeze_762.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf679.data_ptr()))
    del add_71
    del buf661
    del buf665
    del buf675
    del cat_7
    del convolution_28
    del primals_35
    del squeeze_40
    del unsqueeze_762
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf683 = aten.convolution_backward(reinterpret_tensor(buf681, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_81, primals_152, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False])
    del getitem_81
    del primals_152
    buf684 = buf683[0]
    buf685 = buf683[1]
    del buf683
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf686 = aten.convolution_backward(reinterpret_tensor(buf681, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_78, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False])
    del getitem_78
    del primals_151
    buf687 = buf686[0]
    buf688 = buf686[1]
    del buf686
    buf689 = empty((336, ), device='cpu', dtype=torch.float32)
    buf690 = empty((336, ), device='cpu', dtype=torch.float32)
    buf691 = buf681; del buf681  # reuse
    buf692 = buf690; del buf690  # reuse
    cpp_fused_cat_mul_native_batch_norm_backward_83(c_void_p(buf692.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(mul_1068.data_ptr()), c_void_p(cat_6.data_ptr()), c_void_p(unsqueeze_774.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf691.data_ptr()))
    del buf684
    del buf687
    del cat_6
    del mul_1068
    del primals_33
    del squeeze_37
    del unsqueeze_774
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf693 = aten.convolution_backward(reinterpret_tensor(buf691, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_73, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_73
    del primals_150
    buf694 = buf693[0]
    buf695 = buf693[1]
    del buf693
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf696 = aten.convolution_backward(reinterpret_tensor(buf691, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_72, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf691
    del getitem_72
    del primals_149
    buf697 = buf696[0]
    buf698 = buf696[1]
    del buf696
    buf699 = empty((56, ), device='cpu', dtype=torch.float32)
    buf700 = empty((56, ), device='cpu', dtype=torch.float32)
    buf701 = buf654; del buf654  # reuse
    buf702 = buf700; del buf700  # reuse
    cpp_fused_add_cat_native_batch_norm_backward_84(c_void_p(buf701.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_786.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf699.data_ptr()))
    del buf694
    del buf697
    del convolution_22
    del primals_31
    del squeeze_34
    del unsqueeze_786
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf703 = aten.convolution_backward(buf701, mul_80, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf701
    del mul_80
    del primals_148
    buf704 = buf703[0]
    buf705 = buf703[1]
    del buf703
    buf706 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf707 = reinterpret_tensor(buf706, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf706  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_85(c_void_p(buf707.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(add_56.data_ptr()), c_void_p(convolution_21.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf708 = aten.convolution_backward(buf707, mul_79, primals_146, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf707
    del mul_79
    del primals_146
    buf709 = buf708[0]
    buf710 = buf708[1]
    buf711 = buf708[2]
    del buf708
    buf712 = reinterpret_tensor(buf709, (8, 20, 1, 1), (20, 1, 1, 1), 0); del buf709  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_86(c_void_p(buf712.data_ptr()), c_void_p(convolution_20.data_ptr()))
    del convolution_20
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf713 = aten.convolution_backward(buf712, mean, primals_144, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf712
    del mean
    del primals_144
    buf714 = buf713[0]
    buf715 = buf713[1]
    buf716 = buf713[2]
    del buf713
    buf717 = empty((240, ), device='cpu', dtype=torch.float32)
    buf718 = empty((240, ), device='cpu', dtype=torch.float32)
    buf719 = buf704; del buf704  # reuse
    buf720 = buf718; del buf718  # reuse
    buf721 = reinterpret_tensor(buf304, (8, 60, 28, 28), (47040, 1, 1680, 60), 0); del buf304  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_87(c_void_p(buf719.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(add_56.data_ptr()), c_void_p(cat_5.data_ptr()), c_void_p(unsqueeze_798.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf721.data_ptr()))
    del add_56
    del buf714
    del cat_5
    del convolution_21
    del unsqueeze_798
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf722 = aten.convolution_backward(buf721, constant_pad_nd_7, primals_28, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False])
    del constant_pad_nd_7
    del primals_28
    buf723 = buf722[0]
    buf724 = buf722[1]
    del buf722
    buf725 = buf721; del buf721  # reuse
    cpp_fused_convolution_backward_88(c_void_p(buf719.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf725.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf726 = aten.convolution_backward(buf725, constant_pad_nd_6, primals_27, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False])
    del constant_pad_nd_6
    del primals_27
    buf727 = buf726[0]
    buf728 = buf726[1]
    del buf726
    buf729 = buf725; del buf725  # reuse
    cpp_fused_convolution_backward_89(c_void_p(buf719.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf729.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf730 = aten.convolution_backward(buf729, constant_pad_nd_5, primals_26, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False])
    del constant_pad_nd_5
    del primals_26
    buf731 = buf730[0]
    buf732 = buf730[1]
    del buf730
    buf733 = buf729; del buf729  # reuse
    cpp_fused_convolution_backward_90(c_void_p(buf719.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf733.data_ptr()))
    del buf719
    del primals_29
    del squeeze_31
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf734 = aten.convolution_backward(buf733, constant_pad_nd_4, primals_25, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False])
    del buf733
    del constant_pad_nd_4
    del primals_25
    buf735 = buf734[0]
    buf736 = buf734[1]
    del buf734
    buf737 = empty_strided((8, 240, 56, 56), (752640, 1, 13440, 240), device='cpu', dtype=torch.float32)
    buf738 = empty((240, ), device='cpu', dtype=torch.float32)
    buf739 = empty((240, ), device='cpu', dtype=torch.float32)
    buf740 = empty((240, ), device='cpu', dtype=torch.float32)
    buf741 = buf737; del buf737  # reuse
    cpp_fused_cat_convolution_backward_mul_native_batch_norm_backward_91(c_void_p(buf741.data_ptr()), c_void_p(buf735.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(mul_1108.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_810.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf740.data_ptr()))
    del buf723
    del buf727
    del buf731
    del buf735
    del buf739
    del convolution_15
    del mul_1108
    del primals_23
    del squeeze_28
    del unsqueeze_810
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf742 = aten.convolution_backward(buf741, add_46, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_46
    del buf741
    del primals_143
    buf743 = buf742[0]
    buf744 = buf742[1]
    del buf742
    buf745 = empty((40, ), device='cpu', dtype=torch.float32)
    buf746 = empty((40, ), device='cpu', dtype=torch.float32)
    buf747 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cpu', dtype=torch.float32)
    buf748 = buf746; del buf746  # reuse
    cpp_fused_native_batch_norm_backward_92(c_void_p(buf748.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(cat_4.data_ptr()), c_void_p(unsqueeze_822.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf747.data_ptr()))
    del cat_4
    del primals_21
    del squeeze_25
    del unsqueeze_822
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf749 = aten.convolution_backward(reinterpret_tensor(buf747, (8, 20, 56, 56), (125440, 1, 2240, 40), 20), getitem_43, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_43
    del primals_142
    buf750 = buf749[0]
    buf751 = buf749[1]
    del buf749
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf752 = aten.convolution_backward(reinterpret_tensor(buf747, (8, 20, 56, 56), (125440, 1, 2240, 40), 0), getitem_40, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf747
    del getitem_40
    del primals_141
    buf753 = buf752[0]
    buf754 = buf752[1]
    del buf752
    buf755 = empty((120, ), device='cpu', dtype=torch.float32)
    buf756 = empty((120, ), device='cpu', dtype=torch.float32)
    buf757 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    buf758 = buf756; del buf756  # reuse
    cpp_fused_cat_native_batch_norm_backward_threshold_backward_93(c_void_p(buf758.data_ptr()), c_void_p(le_1.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf750.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_834.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf757.data_ptr()))
    del buf750
    del buf753
    del convolution_12
    del le_1
    del primals_19
    del squeeze_22
    del unsqueeze_834
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf759 = aten.convolution_backward(buf757, relu_4, primals_140, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf757
    del primals_140
    buf760 = buf759[0]
    buf761 = buf759[1]
    del buf759
    buf762 = empty((120, ), device='cpu', dtype=torch.float32)
    buf763 = empty((120, ), device='cpu', dtype=torch.float32)
    buf764 = buf760; del buf760  # reuse
    buf765 = buf763; del buf763  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_94(c_void_p(buf764.data_ptr()), c_void_p(buf765.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(cat_3.data_ptr()), c_void_p(unsqueeze_846.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf762.data_ptr()))
    del cat_3
    del primals_17
    del relu_4
    del squeeze_19
    del unsqueeze_846
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf766 = aten.convolution_backward(reinterpret_tensor(buf764, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), getitem_33, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_33
    del primals_139
    buf767 = buf766[0]
    buf768 = buf766[1]
    del buf766
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf769 = aten.convolution_backward(reinterpret_tensor(buf764, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), getitem_32, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf764
    del getitem_32
    del primals_138
    buf770 = buf769[0]
    buf771 = buf769[1]
    del buf769
    buf772 = empty((40, ), device='cpu', dtype=torch.float32)
    buf773 = empty((40, ), device='cpu', dtype=torch.float32)
    buf774 = buf743; del buf743  # reuse
    buf775 = buf773; del buf773  # reuse
    cpp_fused_add_cat_native_batch_norm_backward_95(c_void_p(buf774.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(cat_2.data_ptr()), c_void_p(unsqueeze_858.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf772.data_ptr()))
    del buf767
    del buf770
    del cat_2
    del primals_15
    del squeeze_16
    del unsqueeze_858
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf776 = aten.convolution_backward(reinterpret_tensor(buf774, (8, 20, 56, 56), (125440, 1, 2240, 40), 20), getitem_29, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_29
    del primals_137
    buf777 = buf776[0]
    buf778 = buf776[1]
    del buf776
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf779 = aten.convolution_backward(reinterpret_tensor(buf774, (8, 20, 56, 56), (125440, 1, 2240, 40), 0), getitem_26, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf774
    del getitem_26
    del primals_136
    buf780 = buf779[0]
    buf781 = buf779[1]
    del buf779
    buf782 = empty((192, ), device='cpu', dtype=torch.float32)
    buf783 = empty((192, ), device='cpu', dtype=torch.float32)
    buf784 = empty_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.float32)
    buf785 = buf783; del buf783  # reuse
    cpp_fused_cat_native_batch_norm_backward_threshold_backward_96(c_void_p(buf785.data_ptr()), c_void_p(le_3.data_ptr()), c_void_p(buf780.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(cat_1.data_ptr()), c_void_p(unsqueeze_870.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(buf784.data_ptr()))
    del buf777
    del buf780
    del cat_1
    del le_3
    del primals_13
    del squeeze_13
    del unsqueeze_870
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf786 = aten.convolution_backward(reinterpret_tensor(buf784, (8, 64, 56, 56), (602112, 1, 10752, 192), 128), constant_pad_nd_3, primals_12, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False])
    del constant_pad_nd_3
    del primals_12
    buf787 = buf786[0]
    buf788 = buf786[1]
    del buf786
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf789 = aten.convolution_backward(reinterpret_tensor(buf784, (8, 64, 56, 56), (602112, 1, 10752, 192), 64), constant_pad_nd_2, primals_11, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False])
    del constant_pad_nd_2
    del primals_11
    buf790 = buf789[0]
    buf791 = buf789[1]
    del buf789
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf792 = aten.convolution_backward(reinterpret_tensor(buf784, (8, 64, 56, 56), (602112, 1, 10752, 192), 0), constant_pad_nd_1, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False])
    del buf784
    del constant_pad_nd_1
    del primals_10
    buf793 = buf792[0]
    buf794 = buf792[1]
    del buf792
    buf795 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cpu', dtype=torch.float32)
    buf796 = empty((192, ), device='cpu', dtype=torch.float32)
    buf797 = empty((192, ), device='cpu', dtype=torch.float32)
    buf798 = buf795; del buf795  # reuse
    buf799 = buf797; del buf797  # reuse
    cpp_fused_cat_native_batch_norm_backward_threshold_backward_97(c_void_p(buf798.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(le_4.data_ptr()), c_void_p(cat.data_ptr()), c_void_p(unsqueeze_882.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf796.data_ptr()))
    del buf787
    del buf790
    del buf793
    del cat
    del le_4
    del primals_8
    del squeeze_10
    del unsqueeze_882
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf800 = aten.convolution_backward(reinterpret_tensor(buf798, (8, 96, 112, 112), (2408448, 1, 21504, 192), 96), getitem_7, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del getitem_7
    del primals_135
    buf801 = buf800[0]
    buf802 = buf800[1]
    del buf800
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf803 = aten.convolution_backward(reinterpret_tensor(buf798, (8, 96, 112, 112), (2408448, 1, 21504, 192), 0), getitem_6, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf798
    del getitem_6
    del primals_134
    buf804 = buf803[0]
    buf805 = buf803[1]
    del buf803
    buf806 = empty((32, ), device='cpu', dtype=torch.float32)
    buf807 = empty((32, ), device='cpu', dtype=torch.float32)
    buf808 = empty((32, ), device='cpu', dtype=torch.float32)
    buf809 = empty((8, 32, 112, 112), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_backward_native_batch_norm_backward_98(c_void_p(buf804.data_ptr()), c_void_p(buf801.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_894.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf809.data_ptr()))
    del convolution_2
    del primals_6
    del squeeze_7
    del unsqueeze_894
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.native_batch_norm_backward]
    buf810 = aten.convolution_backward(buf809, relu_1, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf809
    del primals_133
    buf811 = buf810[0]
    buf812 = buf810[1]
    del buf810
    buf813 = buf807; del buf807  # reuse
    buf814 = empty((32, ), device='cpu', dtype=torch.float32)
    buf815 = empty((32, ), device='cpu', dtype=torch.float32)
    buf816 = buf811; del buf811  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99(c_void_p(buf816.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_906.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf815.data_ptr()))
    del convolution_1
    del primals_4
    del relu_1
    del squeeze_4
    del unsqueeze_906
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf817 = aten.convolution_backward(buf816, relu, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf816
    del primals_132
    buf818 = buf817[0]
    buf819 = buf817[1]
    del buf817
    buf820 = buf814; del buf814  # reuse
    buf821 = empty((32, ), device='cpu', dtype=torch.float32)
    buf822 = buf818; del buf818  # reuse
    buf823 = buf821; del buf821  # reuse
    buf824 = buf822; del buf822  # reuse
    cpp_fused_add_cat_convolution_backward_native_batch_norm_backward_threshold_backward_100(c_void_p(buf824.data_ptr()), c_void_p(buf823.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf801.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_918.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf820.data_ptr()))
    del buf801
    del buf804
    del convolution
    del primals_2
    del relu
    del squeeze_1
    del unsqueeze_918
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf825 = aten.convolution_backward(buf824, constant_pad_nd, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf824
    del constant_pad_nd
    del primals_1
    buf826 = buf825[1]
    return (buf826, buf823, buf820, buf815, buf813, buf808, buf806, buf799, buf796, buf794, buf791, buf788, buf785, buf782, buf775, buf772, buf765, buf762, buf758, buf755, buf748, buf745, buf740, buf738, buf736, buf732, buf728, buf724, buf720, buf717, buf702, buf699, buf692, buf689, buf682, buf679, buf658, buf655, buf647, buf644, buf637, buf634, buf615, buf612, buf605, buf602, buf595, buf592, buf573, buf570, buf565, buf563, buf561, buf557, buf553, buf549, buf546, buf531, buf528, buf521, buf518, buf504, buf501, buf480, buf477, buf469, buf466, buf452, buf449, buf430, buf427, buf420, buf417, buf403, buf400, buf381, buf378, buf373, buf371, buf366, buf363, buf348, buf345, buf338, buf335, buf321, buf318, buf297, buf294, buf286, buf283, buf269, buf266, buf247, buf244, buf237, buf234, buf220, buf217, buf198, buf195, buf190, buf188, buf186, buf182, buf178, buf174, buf170, buf167, buf151, buf148, buf143, buf141, buf127, buf124, buf105, buf102, buf97, buf95, buf81, buf78, buf59, buf56, buf51, buf49, buf35, buf32, buf13, buf10, buf5, buf3, buf819, buf812, buf805, buf802, buf781, buf778, buf771, buf768, buf761, buf754, buf751, buf744, buf715, buf716, buf710, buf711, buf705, buf698, buf695, buf688, buf685, buf676, buf677, buf671, buf672, buf666, buf662, buf653, buf650, buf643, buf640, buf631, buf632, buf626, buf627, buf621, buf618, buf611, buf608, buf601, buf598, buf589, buf590, buf584, buf585, buf579, buf576, buf569, buf544, buf545, buf539, buf540, buf534, buf527, buf524, buf516, buf513, buf510, buf507, buf498, buf499, buf493, buf494, buf488, buf484, buf475, buf472, buf464, buf461, buf458, buf455, buf446, buf447, buf441, buf442, buf436, buf433, buf426, buf423, buf415, buf412, buf409, buf406, buf397, buf398, buf392, buf393, buf387, buf384, buf377, buf370, buf361, buf362, buf356, buf357, buf351, buf344, buf341, buf333, buf330, buf327, buf324, buf315, buf316, buf310, buf311, buf305, buf301, buf292, buf289, buf281, buf278, buf275, buf272, buf263, buf264, buf258, buf259, buf253, buf250, buf243, buf240, buf232, buf229, buf226, buf223, buf214, buf215, buf209, buf210, buf204, buf201, buf194, buf165, buf166, buf160, buf161, buf155, buf147, buf139, buf136, buf133, buf130, buf121, buf122, buf116, buf117, buf111, buf108, buf101, buf93, buf90, buf87, buf84, buf75, buf76, buf70, buf71, buf65, buf62, buf55, buf47, buf44, buf41, buf38, buf29, buf30, buf24, buf25, buf19, buf16, buf9, reinterpret_tensor(buf1, (1000, 1536), (1536, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((60, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((60, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((60, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((112, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((112, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((240, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((240, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((20, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((14, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((336, 14, 1, 1), (14, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((104, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((624, 104, 1, 1), (104, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((624, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((52, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((624, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((160, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((264, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((1536, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    constant_pad_nd = rand_strided((8, 3, 225, 225), (151875, 1, 675, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((8, 16, 112, 112), (401408, 12544, 112, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((8, 16, 112, 112), (401408, 12544, 112, 1), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    constant_pad_nd_1 = rand_strided((8, 64, 113, 113), (817216, 1, 7232, 64), device='cpu', dtype=torch.float32)
    constant_pad_nd_2 = rand_strided((8, 64, 115, 115), (846400, 1, 7360, 64), device='cpu', dtype=torch.float32)
    constant_pad_nd_3 = rand_strided((8, 64, 117, 117), (876096, 1, 7488, 64), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_26 = rand_strided((8, 96, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.float32)
    getitem_29 = rand_strided((8, 96, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_32 = rand_strided((8, 20, 56, 56), (125440, 1, 2240, 40), device='cpu', dtype=torch.float32)
    getitem_33 = rand_strided((8, 20, 56, 56), (125440, 1, 2240, 40), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_40 = rand_strided((8, 60, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    getitem_43 = rand_strided((8, 60, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_46 = rand_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 240, 56, 56), (752640, 1, 13440, 240), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    constant_pad_nd_4 = rand_strided((8, 60, 57, 57), (194940, 1, 3420, 60), device='cpu', dtype=torch.float32)
    constant_pad_nd_5 = rand_strided((8, 60, 59, 59), (208860, 1, 3540, 60), device='cpu', dtype=torch.float32)
    constant_pad_nd_6 = rand_strided((8, 60, 61, 61), (223260, 1, 3660, 60), device='cpu', dtype=torch.float32)
    constant_pad_nd_7 = rand_strided((8, 60, 63, 63), (238140, 1, 3780, 60), device='cpu', dtype=torch.float32)
    cat_5 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    add_56 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_79 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_72 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    getitem_73 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    cat_6 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_78 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    getitem_81 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    cat_7 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    add_71 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_104 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    getitem_84 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    getitem_85 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    cat_8 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_88 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    getitem_89 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    cat_9 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_94 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    getitem_97 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    cat_10 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    add_87 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_129 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    getitem_100 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    getitem_101 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    cat_11 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_104 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    getitem_105 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    cat_12 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_110 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    getitem_113 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    cat_13 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    add_103 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_154 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    getitem_116 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    getitem_117 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cpu', dtype=torch.float32)
    cat_14 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    add_109 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    constant_pad_nd_8 = rand_strided((8, 112, 29, 29), (94192, 1, 3248, 112), device='cpu', dtype=torch.float32)
    constant_pad_nd_9 = rand_strided((8, 112, 31, 31), (107632, 1, 3472, 112), device='cpu', dtype=torch.float32)
    constant_pad_nd_10 = rand_strided((8, 112, 33, 33), (121968, 1, 3696, 112), device='cpu', dtype=torch.float32)
    cat_15 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    add_119 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 14, 1, 1), (14, 1, 14, 14), device='cpu', dtype=torch.float32)
    mul_179 = rand_strided((8, 14, 1, 1), (14, 1, 14, 14), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    mul_180 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_138 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    getitem_139 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    cat_16 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_146 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_151 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_156 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_161 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_17 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    add_134 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cpu', dtype=torch.float32)
    mul_204 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cpu', dtype=torch.float32)
    getitem_164 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_165 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_18 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_168 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    getitem_169 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    cat_19 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_176 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_181 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_186 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_191 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_20 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    add_150 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cpu', dtype=torch.float32)
    mul_229 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cpu', dtype=torch.float32)
    getitem_194 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_195 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_21 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_198 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    getitem_199 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    cat_22 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_206 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_211 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_216 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_221 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_23 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    add_166 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cpu', dtype=torch.float32)
    mul_254 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cpu', dtype=torch.float32)
    convolution_81 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cpu', dtype=torch.float32)
    getitem_224 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_225 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_24 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    add_172 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cpu', dtype=torch.float32)
    convolution_84 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    mul_270 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    convolution_85 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    add_182 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cpu', dtype=torch.float32)
    convolution_86 = rand_strided((8, 52, 1, 1), (52, 1, 52, 52), device='cpu', dtype=torch.float32)
    mul_279 = rand_strided((8, 52, 1, 1), (52, 1, 52, 52), device='cpu', dtype=torch.float32)
    convolution_87 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cpu', dtype=torch.float32)
    mul_280 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    convolution_88 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_234 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    getitem_235 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    cat_25 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_242 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_247 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_252 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_257 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_26 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_197 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_95 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    mul_304 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    convolution_96 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    getitem_260 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_261 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_27 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_264 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    getitem_265 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    cat_28 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_272 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_277 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_282 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_287 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_29 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_213 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_105 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    mul_329 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    convolution_106 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    getitem_290 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_291 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_30 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_294 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    getitem_295 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    cat_31 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_302 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_307 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_312 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_317 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_32 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_229 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_115 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    mul_354 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    convolution_116 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    getitem_320 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    getitem_321 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cpu', dtype=torch.float32)
    cat_33 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    add_235 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cpu', dtype=torch.float32)
    convolution_119 = rand_strided((8, 960, 14, 14), (188160, 1, 13440, 960), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    constant_pad_nd_11 = rand_strided((8, 240, 15, 15), (54000, 1, 3600, 240), device='cpu', dtype=torch.float32)
    constant_pad_nd_12 = rand_strided((8, 240, 17, 17), (69360, 1, 4080, 240), device='cpu', dtype=torch.float32)
    constant_pad_nd_13 = rand_strided((8, 240, 19, 19), (86640, 1, 4560, 240), device='cpu', dtype=torch.float32)
    constant_pad_nd_14 = rand_strided((8, 240, 21, 21), (105840, 1, 5040, 240), device='cpu', dtype=torch.float32)
    cat_34 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    add_245 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    mean_12 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    convolution_124 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    mul_379 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    convolution_125 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    mul_380 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_126 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    add_250 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cpu', dtype=torch.float32)
    convolution_127 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_356 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_361 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_366 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_371 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    cat_35 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    add_260 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    mean_13 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cpu', dtype=torch.float32)
    convolution_132 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cpu', dtype=torch.float32)
    mul_404 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cpu', dtype=torch.float32)
    convolution_133 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cpu', dtype=torch.float32)
    getitem_374 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_375 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    cat_36 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    add_266 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cpu', dtype=torch.float32)
    convolution_136 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_384 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_389 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_394 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_399 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    cat_37 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    add_276 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    mean_14 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cpu', dtype=torch.float32)
    convolution_141 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cpu', dtype=torch.float32)
    mul_429 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cpu', dtype=torch.float32)
    convolution_142 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cpu', dtype=torch.float32)
    getitem_402 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_403 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    cat_38 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    add_282 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cpu', dtype=torch.float32)
    convolution_145 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    getitem_412 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_417 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_422 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_427 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    cat_39 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    add_292 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    mean_15 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cpu', dtype=torch.float32)
    convolution_150 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cpu', dtype=torch.float32)
    mul_454 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cpu', dtype=torch.float32)
    convolution_151 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cpu', dtype=torch.float32)
    getitem_430 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    getitem_431 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cpu', dtype=torch.float32)
    cat_40 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    add_298 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cpu', dtype=torch.float32)
    convolution_154 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view = rand_strided((8, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.bool)
    unsqueeze_234 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_508 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_548 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_588 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cpu', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_628 = rand_strided((8, 960, 14, 14), (188160, 1, 13440, 960), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_668 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_708 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_748 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_788 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_828 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_868 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_908 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_948 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_988 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1028 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1068 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_786 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_798 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1108 = rand_strided((8, 240, 56, 56), (752640, 1, 13440, 240), device='cpu', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_822 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_1 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.bool)
    unsqueeze_834 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_858 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_3 = rand_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.bool)
    unsqueeze_870 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_4 = rand_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cpu', dtype=torch.bool)
    unsqueeze_882 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_894 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_906 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_918 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_12, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_54, primals_55, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_236, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, constant_pad_nd, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, getitem_6, getitem_7, cat, squeeze_10, constant_pad_nd_1, constant_pad_nd_2, constant_pad_nd_3, cat_1, squeeze_13, getitem_26, getitem_29, cat_2, squeeze_16, getitem_32, getitem_33, cat_3, squeeze_19, relu_4, convolution_12, squeeze_22, getitem_40, getitem_43, cat_4, squeeze_25, add_46, convolution_15, squeeze_28, constant_pad_nd_4, constant_pad_nd_5, constant_pad_nd_6, constant_pad_nd_7, cat_5, squeeze_31, add_56, mean, convolution_20, mul_79, convolution_21, mul_80, convolution_22, squeeze_34, getitem_72, getitem_73, cat_6, squeeze_37, getitem_78, getitem_81, cat_7, squeeze_40, add_71, mean_1, convolution_27, mul_104, convolution_28, getitem_84, getitem_85, cat_8, squeeze_43, getitem_88, getitem_89, cat_9, squeeze_46, getitem_94, getitem_97, cat_10, squeeze_49, add_87, mean_2, convolution_35, mul_129, convolution_36, getitem_100, getitem_101, cat_11, squeeze_52, getitem_104, getitem_105, cat_12, squeeze_55, getitem_110, getitem_113, cat_13, squeeze_58, add_103, mean_3, convolution_43, mul_154, convolution_44, getitem_116, getitem_117, cat_14, squeeze_61, add_109, convolution_47, squeeze_64, constant_pad_nd_8, constant_pad_nd_9, constant_pad_nd_10, cat_15, squeeze_67, add_119, mean_4, convolution_51, mul_179, convolution_52, mul_180, convolution_53, squeeze_70, getitem_138, getitem_139, cat_16, squeeze_73, getitem_146, getitem_151, getitem_156, getitem_161, cat_17, squeeze_76, add_134, mean_5, convolution_60, mul_204, convolution_61, getitem_164, getitem_165, cat_18, squeeze_79, getitem_168, getitem_169, cat_19, squeeze_82, getitem_176, getitem_181, getitem_186, getitem_191, cat_20, squeeze_85, add_150, mean_6, convolution_70, mul_229, convolution_71, getitem_194, getitem_195, cat_21, squeeze_88, getitem_198, getitem_199, cat_22, squeeze_91, getitem_206, getitem_211, getitem_216, getitem_221, cat_23, squeeze_94, add_166, mean_7, convolution_80, mul_254, convolution_81, getitem_224, getitem_225, cat_24, squeeze_97, add_172, convolution_84, squeeze_100, mul_270, convolution_85, squeeze_103, add_182, mean_8, convolution_86, mul_279, convolution_87, mul_280, convolution_88, squeeze_106, getitem_234, getitem_235, cat_25, squeeze_109, getitem_242, getitem_247, getitem_252, getitem_257, cat_26, squeeze_112, add_197, mean_9, convolution_95, mul_304, convolution_96, getitem_260, getitem_261, cat_27, squeeze_115, getitem_264, getitem_265, cat_28, squeeze_118, getitem_272, getitem_277, getitem_282, getitem_287, cat_29, squeeze_121, add_213, mean_10, convolution_105, mul_329, convolution_106, getitem_290, getitem_291, cat_30, squeeze_124, getitem_294, getitem_295, cat_31, squeeze_127, getitem_302, getitem_307, getitem_312, getitem_317, cat_32, squeeze_130, add_229, mean_11, convolution_115, mul_354, convolution_116, getitem_320, getitem_321, cat_33, squeeze_133, add_235, convolution_119, squeeze_136, constant_pad_nd_11, constant_pad_nd_12, constant_pad_nd_13, constant_pad_nd_14, cat_34, squeeze_139, add_245, mean_12, convolution_124, mul_379, convolution_125, mul_380, convolution_126, squeeze_142, add_250, convolution_127, squeeze_145, getitem_356, getitem_361, getitem_366, getitem_371, cat_35, squeeze_148, add_260, mean_13, convolution_132, mul_404, convolution_133, getitem_374, getitem_375, cat_36, squeeze_151, add_266, convolution_136, squeeze_154, getitem_384, getitem_389, getitem_394, getitem_399, cat_37, squeeze_157, add_276, mean_14, convolution_141, mul_429, convolution_142, getitem_402, getitem_403, cat_38, squeeze_160, add_282, convolution_145, squeeze_163, getitem_412, getitem_417, getitem_422, getitem_427, cat_39, squeeze_166, add_292, mean_15, convolution_150, mul_454, convolution_151, getitem_430, getitem_431, cat_40, squeeze_169, add_298, convolution_154, squeeze_172, view, permute_1, le, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_508, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_548, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_588, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_628, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_668, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_708, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_748, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_788, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_828, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_868, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_908, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_948, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_988, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1028, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1068, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1108, unsqueeze_810, unsqueeze_822, le_1, unsqueeze_834, unsqueeze_846, unsqueeze_858, le_3, unsqueeze_870, le_4, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_mixnet_l', benchmark_compiled_module)
