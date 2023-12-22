
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


cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1280L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1280L*x2) + (46080L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1280L*x2) + (46080L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(36.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp9 = tmp5 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(8L))
                    {
                        float tmp24[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1280L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (46080L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (46080L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp1 = static_cast<float>(36.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(0.003472222222222222);
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
                            tmp23.store(tmp24 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp24, 8, out_ptr4 + static_cast<long>(x1 + (36L*x2) + (46080L*x0)), static_cast<long>(36L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (1280L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (1280L*x1) + (46080L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (1280L*x1) + (46080L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(x2)];
                        auto tmp8 = out_ptr2[static_cast<long>(x2)];
                        auto tmp11 = in_ptr5[static_cast<long>(x2)];
                        auto tmp16 = out_ptr1[static_cast<long>(x2)];
                        auto tmp19 = in_ptr6[static_cast<long>(x2)];
                        auto tmp1 = static_cast<float>(36.0);
                        auto tmp2 = tmp0 / tmp1;
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp9 = static_cast<float>(0.003472222222222222);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp12 = decltype(tmp11)(tmp11 * tmp11);
                        auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                        auto tmp14 = decltype(tmp7)(tmp7 * tmp13);
                        auto tmp15 = decltype(tmp4)(tmp4 - tmp14);
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp9);
                        auto tmp18 = decltype(tmp15)(tmp15 - tmp17);
                        auto tmp20 = decltype(tmp11)(tmp11 * tmp19);
                        auto tmp21 = decltype(tmp18)(tmp18 * tmp20);
                        out_ptr4[static_cast<long>(x1 + (36L*x2) + (46080L*x0))] = tmp21;
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
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
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
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
                auto tmp5 = static_cast<float>(0.003472222222222222);
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
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_2 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(36.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(36.0);
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
                        auto tmp22 = static_cast<float>(0.003472222222222222);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.003472222222222222);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_6 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.003472222222222222);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(36.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(36.0);
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
                        auto tmp22 = static_cast<float>(0.003472222222222222);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.003472222222222222);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_11 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.003472222222222222);
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
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_12 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(36.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(36.0);
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
                        auto tmp22 = static_cast<float>(0.003472222222222222);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.003472222222222222);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_16 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp7 = tmp5 - tmp6;
                auto tmp9 = static_cast<float>(0.003472222222222222);
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
                tmp22.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(36.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(36.0);
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
                        auto tmp22 = static_cast<float>(0.003472222222222222);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.003472222222222222);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_21 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp9 = tmp7 - tmp8;
                auto tmp11 = static_cast<float>(0.003472222222222222);
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
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
            }
        }
    }
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
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_22 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x2) + (41472L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x2) + (41472L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(36.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(36.0);
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
                        auto tmp22 = static_cast<float>(0.003472222222222222);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (41472L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.003472222222222222);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_26 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(55296L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.003472222222222222);
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_27 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x2) + (24192L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x2) + (24192L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_28 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(36L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x2) + (24192L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x2) + (24192L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (672L*x2) + (24192L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(36.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (24192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (24192L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (672L*x1) + (24192L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(36.0);
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
                        auto tmp22 = static_cast<float>(0.003472222222222222);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (24192L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (112L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0008680555555555555);
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_32 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x2) + (96768L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x2) + (96768L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_33 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x2) + (96768L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x2) + (96768L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (672L*x2) + (96768L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(144.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(144.0);
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
                        auto tmp22 = static_cast<float>(0.0008680555555555555);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (112L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_37 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x2) + (96768L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x2) + (96768L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_38 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x2) + (96768L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x2) + (96768L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (672L*x2) + (96768L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(144.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(144.0);
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
                        auto tmp22 = static_cast<float>(0.0008680555555555555);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (112L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.0008680555555555555);
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
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_42 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x2) + (96768L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x2) + (96768L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_43 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x2) + (96768L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x2) + (96768L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (672L*x2) + (96768L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(144.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(144.0);
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
                        auto tmp22 = static_cast<float>(0.0008680555555555555);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (96768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (112L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (112L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp11 = static_cast<float>(0.0008680555555555555);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_47 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_48 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_49 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(144.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(144.0);
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
                        auto tmp22 = static_cast<float>(0.0008680555555555555);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_51 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.0008680555555555555);
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
                tmp18.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_52 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_53 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_54 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(144.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(144.0);
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
                        auto tmp22 = static_cast<float>(0.0008680555555555555);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_56 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                tmp20.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_57 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_58 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_59 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(144.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(144.0);
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
                        auto tmp22 = static_cast<float>(0.0008680555555555555);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_61 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (80L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (80L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp7 = tmp5 - tmp6;
                auto tmp9 = static_cast<float>(0.0008680555555555555);
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
                tmp22.store(out_ptr2 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x2) + (69120L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
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
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (69120L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(144.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(144.0);
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
                        auto tmp22 = static_cast<float>(0.0008680555555555555);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (69120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0008680555555555555);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_66 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (80L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (80L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (80L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp9 = tmp7 - tmp8;
                auto tmp11 = static_cast<float>(0.0008680555555555555);
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
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_67 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x2) + (34560L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x2) + (34560L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x2) + (34560L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x2) + (34560L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (240L*x2) + (34560L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(144.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (34560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (34560L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (240L*x1) + (34560L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(144.0);
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
                        auto tmp22 = static_cast<float>(0.0008680555555555555);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (34560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4608L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00021701388888888888);
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


cpp_fused_convolution_backward_native_batch_norm_backward_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4608L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
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
                    auto tmp5 = static_cast<float>(0.00021701388888888888);
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_72 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x2) + (138240L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x2) + (138240L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_74 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x2) + (138240L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x2) + (138240L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (240L*x2) + (138240L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(576.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (138240L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (138240L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (240L*x1) + (138240L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(576.0);
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
                        auto tmp22 = static_cast<float>(0.00021701388888888888);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (138240L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4608L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00021701388888888888);
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4608L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00021701388888888888);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_77 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (144L*x2) + (82944L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x2) + (82944L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(576L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x2) + (82944L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (144L*x2) + (82944L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (144L*x2) + (82944L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(576.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (82944L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (82944L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (144L*x1) + (82944L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(576.0);
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
                        auto tmp22 = static_cast<float>(0.00021701388888888888);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (82944L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(18432L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(5.425347222222222e-05);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_81 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(18432L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
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
                    auto tmp5 = static_cast<float>(5.425347222222222e-05);
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_82 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (144L*x2) + (331776L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x2) + (331776L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x2) + (331776L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (144L*x2) + (331776L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (144L*x2) + (331776L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(2304.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (331776L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (331776L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (144L*x1) + (331776L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(2304.0);
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
                        auto tmp22 = static_cast<float>(5.425347222222222e-05);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (331776L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(18432L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(5.425347222222222e-05);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_86 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(18432L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(5.425347222222222e-05);
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_87 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x2) + (221184L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x2) + (221184L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_89 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x2) + (221184L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (96L*x2) + (221184L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (96L*x2) + (221184L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(2304.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (96L*x1) + (221184L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (96L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (96L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (96L*x1) + (221184L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (96L*x1) + (221184L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(2304.0);
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
                        auto tmp22 = static_cast<float>(5.425347222222222e-05);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (96L*x1) + (221184L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18432L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(73728L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(73728L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(1.3563368055555555e-05);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_91 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(73728L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(73728L); x0+=static_cast<long>(1L))
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
                    auto tmp5 = static_cast<float>(1.3563368055555555e-05);
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_92 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9216L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (294912L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x2) + (294912L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9216L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x2) + (294912L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (32L*x2) + (294912L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (32L*x2) + (294912L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(9216.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9216L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (32L*x1) + (294912L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (32L*x1) + (294912L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (32L*x1) + (294912L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(9216.0);
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
                        auto tmp22 = static_cast<float>(1.3563368055555555e-05);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (32L*x1) + (294912L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(73728L); x0+=static_cast<long>(1L))
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


cpp_fused_convolution_backward_mul_native_batch_norm_backward_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(73728L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(73728L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(1.3563368055555555e-05);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_118, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_128, primals_130, primals_131, primals_132, primals_133, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_144, primals_145, primals_146, primals_147, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_161, primals_163, primals_165, primals_166, primals_167, primals_168, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_187, primals_188, primals_189, primals_191, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_233, primals_235, primals_236, primals_237, primals_238, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_249, primals_250, primals_427, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, add_9, mean, convolution_2, mul_16, convolution_3, mul_17, convolution_4, squeeze_7, add_14, convolution_5, squeeze_10, mul_32, convolution_6, squeeze_13, add_24, mean_1, convolution_7, mul_41, convolution_8, mul_42, convolution_9, squeeze_16, add_29, convolution_10, squeeze_19, mul_57, convolution_11, squeeze_22, add_39, mean_2, convolution_12, mul_66, convolution_13, mul_67, convolution_14, squeeze_25, add_45, convolution_15, squeeze_28, mul_82, convolution_16, squeeze_31, add_55, mean_3, convolution_17, mul_91, convolution_18, mul_92, convolution_19, squeeze_34, add_60, convolution_20, squeeze_37, mul_107, convolution_21, squeeze_40, add_70, mean_4, convolution_22, mul_116, convolution_23, mul_117, convolution_24, squeeze_43, add_76, convolution_25, squeeze_46, mul_132, convolution_26, squeeze_49, add_86, mean_5, convolution_27, mul_141, convolution_28, mul_142, convolution_29, squeeze_52, add_91, convolution_30, squeeze_55, mul_157, convolution_31, squeeze_58, add_101, mean_6, convolution_32, mul_166, convolution_33, mul_167, convolution_34, squeeze_61, add_107, convolution_35, squeeze_64, mul_182, convolution_36, squeeze_67, add_117, mean_7, convolution_37, mul_191, convolution_38, mul_192, convolution_39, squeeze_70, add_123, convolution_40, squeeze_73, mul_207, convolution_41, squeeze_76, add_133, mean_8, convolution_42, mul_216, convolution_43, mul_217, convolution_44, squeeze_79, add_139, convolution_45, squeeze_82, mul_232, convolution_46, squeeze_85, add_149, mean_9, convolution_47, mul_241, convolution_48, mul_242, convolution_49, squeeze_88, add_154, convolution_50, squeeze_91, mul_257, convolution_51, squeeze_94, add_164, mean_10, convolution_52, mul_266, convolution_53, mul_267, convolution_54, squeeze_97, add_170, convolution_55, squeeze_100, mul_282, convolution_56, squeeze_103, add_180, mean_11, convolution_57, mul_291, convolution_58, mul_292, convolution_59, squeeze_106, add_186, convolution_60, squeeze_109, mul_307, convolution_61, squeeze_112, add_196, mean_12, convolution_62, mul_316, convolution_63, mul_317, convolution_64, squeeze_115, add_202, convolution_65, squeeze_118, mul_332, convolution_66, squeeze_121, add_212, mean_13, convolution_67, mul_341, convolution_68, mul_342, convolution_69, squeeze_124, add_217, convolution_70, squeeze_127, mul_357, convolution_71, squeeze_130, add_227, mean_14, convolution_72, mul_366, convolution_73, mul_367, convolution_74, squeeze_133, add_233, convolution_75, squeeze_136, mul_382, convolution_76, squeeze_139, add_243, mean_15, convolution_77, mul_391, convolution_78, mul_392, convolution_79, squeeze_142, add_249, convolution_80, squeeze_145, mul_407, convolution_81, squeeze_148, add_259, mean_16, convolution_82, mul_416, convolution_83, mul_417, convolution_84, squeeze_151, add_265, convolution_85, squeeze_154, mul_432, convolution_86, squeeze_157, add_275, mean_17, convolution_87, mul_441, convolution_88, mul_442, convolution_89, squeeze_160, add_281, convolution_90, squeeze_163, mul_457, convolution_91, squeeze_166, add_291, mean_18, convolution_92, mul_466, convolution_93, mul_467, convolution_94, squeeze_169, add_296, convolution_95, squeeze_172, view, permute_1, mul_484, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_524, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_564, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_604, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_644, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_684, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_724, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_764, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_804, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_844, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_884, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_924, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_964, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_1004, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1044, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1084, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1124, unsqueeze_810, unsqueeze_822, unsqueeze_834, mul_1164, unsqueeze_846, unsqueeze_858, unsqueeze_870, mul_1204, unsqueeze_882, unsqueeze_894, unsqueeze_906, mul_1244, unsqueeze_918, tangents_1 = args
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
    assert_size_stride(primals_23, (40, ), (1, ))
    assert_size_stride(primals_25, (240, ), (1, ))
    assert_size_stride(primals_27, (240, ), (1, ))
    assert_size_stride(primals_29, (40, ), (1, ))
    assert_size_stride(primals_31, (240, ), (1, ))
    assert_size_stride(primals_33, (240, ), (1, ))
    assert_size_stride(primals_35, (80, ), (1, ))
    assert_size_stride(primals_37, (480, ), (1, ))
    assert_size_stride(primals_39, (480, ), (1, ))
    assert_size_stride(primals_41, (80, ), (1, ))
    assert_size_stride(primals_43, (480, ), (1, ))
    assert_size_stride(primals_45, (480, ), (1, ))
    assert_size_stride(primals_47, (80, ), (1, ))
    assert_size_stride(primals_49, (480, ), (1, ))
    assert_size_stride(primals_51, (480, ), (1, ))
    assert_size_stride(primals_53, (80, ), (1, ))
    assert_size_stride(primals_55, (480, ), (1, ))
    assert_size_stride(primals_57, (480, ), (1, ))
    assert_size_stride(primals_59, (112, ), (1, ))
    assert_size_stride(primals_61, (672, ), (1, ))
    assert_size_stride(primals_63, (672, ), (1, ))
    assert_size_stride(primals_65, (112, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_69, (672, ), (1, ))
    assert_size_stride(primals_71, (112, ), (1, ))
    assert_size_stride(primals_73, (672, ), (1, ))
    assert_size_stride(primals_75, (672, ), (1, ))
    assert_size_stride(primals_77, (112, ), (1, ))
    assert_size_stride(primals_79, (672, ), (1, ))
    assert_size_stride(primals_81, (672, ), (1, ))
    assert_size_stride(primals_83, (192, ), (1, ))
    assert_size_stride(primals_85, (1152, ), (1, ))
    assert_size_stride(primals_87, (1152, ), (1, ))
    assert_size_stride(primals_89, (192, ), (1, ))
    assert_size_stride(primals_91, (1152, ), (1, ))
    assert_size_stride(primals_93, (1152, ), (1, ))
    assert_size_stride(primals_95, (192, ), (1, ))
    assert_size_stride(primals_97, (1152, ), (1, ))
    assert_size_stride(primals_99, (1152, ), (1, ))
    assert_size_stride(primals_101, (192, ), (1, ))
    assert_size_stride(primals_103, (1152, ), (1, ))
    assert_size_stride(primals_105, (1152, ), (1, ))
    assert_size_stride(primals_107, (192, ), (1, ))
    assert_size_stride(primals_109, (1152, ), (1, ))
    assert_size_stride(primals_111, (1152, ), (1, ))
    assert_size_stride(primals_113, (320, ), (1, ))
    assert_size_stride(primals_115, (1280, ), (1, ))
    assert_size_stride(primals_117, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_118, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_121, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_123, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_124, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_125, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_126, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_128, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_130, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_131, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_132, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_135, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_137, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_138, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_139, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_140, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_142, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_144, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_145, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_146, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_147, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_149, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_151, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_152, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_153, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_154, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_156, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_158, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_159, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_160, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_163, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_165, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_166, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_167, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_168, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_170, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_172, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_173, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_174, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_175, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_177, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_179, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_180, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_181, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_182, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_184, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_186, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_187, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_188, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_189, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_191, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_193, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_194, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_195, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_198, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_200, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_201, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_202, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_203, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_205, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_207, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_208, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_209, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_210, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_212, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_214, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_215, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_216, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_217, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_219, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_221, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_222, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_223, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_224, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_226, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_228, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_229, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_230, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_231, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_233, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_235, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_236, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_237, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_238, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_240, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_242, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_243, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_244, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_245, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_247, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_249, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_250, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_427, (8, 3, 192, 192), (110592, 1, 576, 3))
    assert_size_stride(convolution, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(mul_7, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(convolution_1, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(add_9, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(mean, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(convolution_2, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(mul_16, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_3, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(mul_17, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(convolution_4, (8, 16, 96, 96), (147456, 1, 1536, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_14, (8, 16, 96, 96), (147456, 1, 1536, 16))
    assert_size_stride(convolution_5, (8, 96, 96, 96), (884736, 1, 9216, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(mul_32, (8, 96, 96, 96), (884736, 1, 9216, 96))
    assert_size_stride(convolution_6, (8, 96, 48, 48), (221184, 1, 4608, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(add_24, (8, 96, 48, 48), (221184, 1, 4608, 96))
    assert_size_stride(mean_1, (8, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(convolution_7, (8, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(mul_41, (8, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(convolution_8, (8, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(mul_42, (8, 96, 48, 48), (221184, 1, 4608, 96))
    assert_size_stride(convolution_9, (8, 24, 48, 48), (55296, 1, 1152, 24))
    assert_size_stride(squeeze_16, (24, ), (1, ))
    assert_size_stride(add_29, (8, 24, 48, 48), (55296, 1, 1152, 24))
    assert_size_stride(convolution_10, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(squeeze_19, (144, ), (1, ))
    assert_size_stride(mul_57, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(convolution_11, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(squeeze_22, (144, ), (1, ))
    assert_size_stride(add_39, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(mean_2, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_12, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_66, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_13, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_67, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(convolution_14, (8, 24, 48, 48), (55296, 1, 1152, 24))
    assert_size_stride(squeeze_25, (24, ), (1, ))
    assert_size_stride(add_45, (8, 24, 48, 48), (55296, 1, 1152, 24))
    assert_size_stride(convolution_15, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(squeeze_28, (144, ), (1, ))
    assert_size_stride(mul_82, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(convolution_16, (8, 144, 24, 24), (82944, 1, 3456, 144))
    assert_size_stride(squeeze_31, (144, ), (1, ))
    assert_size_stride(add_55, (8, 144, 24, 24), (82944, 1, 3456, 144))
    assert_size_stride(mean_3, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_17, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_91, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_18, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_92, (8, 144, 24, 24), (82944, 1, 3456, 144))
    assert_size_stride(convolution_19, (8, 40, 24, 24), (23040, 1, 960, 40))
    assert_size_stride(squeeze_34, (40, ), (1, ))
    assert_size_stride(add_60, (8, 40, 24, 24), (23040, 1, 960, 40))
    assert_size_stride(convolution_20, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(squeeze_37, (240, ), (1, ))
    assert_size_stride(mul_107, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(convolution_21, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(squeeze_40, (240, ), (1, ))
    assert_size_stride(add_70, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(mean_4, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_22, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_116, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_23, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_117, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(convolution_24, (8, 40, 24, 24), (23040, 1, 960, 40))
    assert_size_stride(squeeze_43, (40, ), (1, ))
    assert_size_stride(add_76, (8, 40, 24, 24), (23040, 1, 960, 40))
    assert_size_stride(convolution_25, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(squeeze_46, (240, ), (1, ))
    assert_size_stride(mul_132, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(convolution_26, (8, 240, 12, 12), (34560, 1, 2880, 240))
    assert_size_stride(squeeze_49, (240, ), (1, ))
    assert_size_stride(add_86, (8, 240, 12, 12), (34560, 1, 2880, 240))
    assert_size_stride(mean_5, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_27, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_141, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_28, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_142, (8, 240, 12, 12), (34560, 1, 2880, 240))
    assert_size_stride(convolution_29, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(squeeze_52, (80, ), (1, ))
    assert_size_stride(add_91, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(convolution_30, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_55, (480, ), (1, ))
    assert_size_stride(mul_157, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_31, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_58, (480, ), (1, ))
    assert_size_stride(add_101, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(mean_6, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_32, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_166, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_33, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_167, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_34, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(squeeze_61, (80, ), (1, ))
    assert_size_stride(add_107, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(convolution_35, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_64, (480, ), (1, ))
    assert_size_stride(mul_182, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_36, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_67, (480, ), (1, ))
    assert_size_stride(add_117, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(mean_7, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_37, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_191, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_38, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_192, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_39, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(squeeze_70, (80, ), (1, ))
    assert_size_stride(add_123, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(convolution_40, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_73, (480, ), (1, ))
    assert_size_stride(mul_207, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_41, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_76, (480, ), (1, ))
    assert_size_stride(add_133, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(mean_8, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_42, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_216, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_43, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_217, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_44, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(squeeze_79, (80, ), (1, ))
    assert_size_stride(add_139, (8, 80, 12, 12), (11520, 1, 960, 80))
    assert_size_stride(convolution_45, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_82, (480, ), (1, ))
    assert_size_stride(mul_232, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_46, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(squeeze_85, (480, ), (1, ))
    assert_size_stride(add_149, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(mean_9, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_47, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_241, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_48, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_242, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(convolution_49, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(squeeze_88, (112, ), (1, ))
    assert_size_stride(add_154, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(convolution_50, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_91, (672, ), (1, ))
    assert_size_stride(mul_257, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_51, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_94, (672, ), (1, ))
    assert_size_stride(add_164, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(mean_10, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_52, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_266, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_53, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_267, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_54, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(squeeze_97, (112, ), (1, ))
    assert_size_stride(add_170, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(convolution_55, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_100, (672, ), (1, ))
    assert_size_stride(mul_282, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_56, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_103, (672, ), (1, ))
    assert_size_stride(add_180, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(mean_11, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_57, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_291, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_58, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_292, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_59, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(squeeze_106, (112, ), (1, ))
    assert_size_stride(add_186, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(convolution_60, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_109, (672, ), (1, ))
    assert_size_stride(mul_307, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_61, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_112, (672, ), (1, ))
    assert_size_stride(add_196, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(mean_12, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_62, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_316, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_63, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_317, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_64, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(squeeze_115, (112, ), (1, ))
    assert_size_stride(add_202, (8, 112, 12, 12), (16128, 1, 1344, 112))
    assert_size_stride(convolution_65, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(squeeze_118, (672, ), (1, ))
    assert_size_stride(mul_332, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(convolution_66, (8, 672, 6, 6), (24192, 1, 4032, 672))
    assert_size_stride(squeeze_121, (672, ), (1, ))
    assert_size_stride(add_212, (8, 672, 6, 6), (24192, 1, 4032, 672))
    assert_size_stride(mean_13, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_67, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_341, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_68, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_342, (8, 672, 6, 6), (24192, 1, 4032, 672))
    assert_size_stride(convolution_69, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_124, (192, ), (1, ))
    assert_size_stride(add_217, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_70, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_127, (1152, ), (1, ))
    assert_size_stride(mul_357, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_71, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_130, (1152, ), (1, ))
    assert_size_stride(add_227, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_14, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_72, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_366, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_73, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_367, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_74, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_133, (192, ), (1, ))
    assert_size_stride(add_233, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_75, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_136, (1152, ), (1, ))
    assert_size_stride(mul_382, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_76, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_139, (1152, ), (1, ))
    assert_size_stride(add_243, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_15, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_77, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_391, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_78, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_392, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_79, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_142, (192, ), (1, ))
    assert_size_stride(add_249, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_80, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_145, (1152, ), (1, ))
    assert_size_stride(mul_407, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_81, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_148, (1152, ), (1, ))
    assert_size_stride(add_259, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_16, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_82, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_416, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_83, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_417, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_84, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_151, (192, ), (1, ))
    assert_size_stride(add_265, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_85, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_154, (1152, ), (1, ))
    assert_size_stride(mul_432, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_86, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_157, (1152, ), (1, ))
    assert_size_stride(add_275, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_17, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_87, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_441, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_88, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_442, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_89, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(squeeze_160, (192, ), (1, ))
    assert_size_stride(add_281, (8, 192, 6, 6), (6912, 1, 1152, 192))
    assert_size_stride(convolution_90, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_163, (1152, ), (1, ))
    assert_size_stride(mul_457, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_91, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(squeeze_166, (1152, ), (1, ))
    assert_size_stride(add_291, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(mean_18, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_92, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_466, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_93, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_467, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(convolution_94, (8, 320, 6, 6), (11520, 1, 1920, 320))
    assert_size_stride(squeeze_169, (320, ), (1, ))
    assert_size_stride(add_296, (8, 320, 6, 6), (11520, 1, 1920, 320))
    assert_size_stride(convolution_95, (8, 1280, 6, 6), (46080, 1, 7680, 1280))
    assert_size_stride(squeeze_172, (1280, ), (1, ))
    assert_size_stride(view, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(mul_484, (8, 1280, 6, 6), (46080, 1, 7680, 1280))
    assert_size_stride(unsqueeze_234, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_246, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_524, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_270, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_564, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_306, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_318, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_604, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_342, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_644, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_378, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_684, (8, 1152, 6, 6), (41472, 1, 6912, 1152))
    assert_size_stride(unsqueeze_414, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_724, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(unsqueeze_450, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_764, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(unsqueeze_486, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_804, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(unsqueeze_522, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_844, (8, 672, 12, 12), (96768, 1, 8064, 672))
    assert_size_stride(unsqueeze_558, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_884, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(unsqueeze_594, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_924, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(unsqueeze_630, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_964, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(unsqueeze_666, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_1004, (8, 480, 12, 12), (69120, 1, 5760, 480))
    assert_size_stride(unsqueeze_702, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_1044, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(unsqueeze_738, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_1084, (8, 240, 24, 24), (138240, 1, 5760, 240))
    assert_size_stride(unsqueeze_774, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_786, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_798, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(mul_1124, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(unsqueeze_810, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_822, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(mul_1164, (8, 144, 48, 48), (331776, 1, 6912, 144))
    assert_size_stride(unsqueeze_846, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_858, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_870, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_1204, (8, 96, 96, 96), (884736, 1, 9216, 96))
    assert_size_stride(unsqueeze_882, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_894, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_906, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_1244, (8, 32, 96, 96), (294912, 1, 3072, 32))
    assert_size_stride(unsqueeze_918, (1, 32, 1, 1), (32, 1, 1, 1))
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
    buf6 = empty((8, 1280, 6, 6), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mul_484.data_ptr()), c_void_p(convolution_95.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del buf4
    del convolution_95
    del mul_484
    del primals_115
    del squeeze_172
    del tangents_1
    del unsqueeze_234
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
    buf7 = aten.convolution_backward(buf6, add_296, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_296
    del buf6
    del primals_250
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((320, ), device='cpu', dtype=torch.float32)
    buf11 = empty((320, ), device='cpu', dtype=torch.float32)
    buf12 = empty((320, ), device='cpu', dtype=torch.float32)
    buf13 = buf8; del buf8  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_1(c_void_p(buf13.data_ptr()), c_void_p(convolution_94.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del buf11
    del convolution_94
    del primals_113
    del squeeze_169
    del unsqueeze_246
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf14 = aten.convolution_backward(buf13, mul_467, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_467
    del primals_249
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cpu', dtype=torch.float32)
    buf18 = reinterpret_tensor(buf17, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf17  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_2(c_void_p(buf18.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(add_291.data_ptr()), c_void_p(convolution_93.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf19 = aten.convolution_backward(buf18, mul_466, primals_247, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf18
    del mul_466
    del primals_247
    buf20 = buf19[0]
    buf21 = buf19[1]
    buf22 = buf19[2]
    del buf19
    buf23 = reinterpret_tensor(buf20, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf20  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_3(c_void_p(buf23.data_ptr()), c_void_p(convolution_92.data_ptr()))
    del convolution_92
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf24 = aten.convolution_backward(buf23, mean_18, primals_245, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf23
    del mean_18
    del primals_245
    buf25 = buf24[0]
    buf26 = buf24[1]
    buf27 = buf24[2]
    del buf24
    buf28 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf29 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf30 = buf15; del buf15  # reuse
    buf31 = buf29; del buf29  # reuse
    buf32 = buf30; del buf30  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_4(c_void_p(buf32.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(convolution_93.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(add_291.data_ptr()), c_void_p(convolution_91.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf28.data_ptr()))
    del add_291
    del convolution_91
    del convolution_93
    del primals_111
    del squeeze_166
    del unsqueeze_258
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf33 = aten.convolution_backward(buf32, mul_457, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf32
    del mul_457
    del primals_244
    buf34 = buf33[0]
    buf35 = buf33[1]
    del buf33
    buf36 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf37 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf38 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf39 = buf34; del buf34  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_5(c_void_p(buf39.data_ptr()), c_void_p(mul_524.data_ptr()), c_void_p(convolution_90.data_ptr()), c_void_p(unsqueeze_270.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    del convolution_90
    del mul_524
    del primals_109
    del squeeze_163
    del unsqueeze_270
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf40 = aten.convolution_backward(buf39, add_281, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_281
    del buf39
    del primals_243
    buf41 = buf40[0]
    buf42 = buf40[1]
    del buf40
    buf43 = empty((192, ), device='cpu', dtype=torch.float32)
    buf44 = empty((192, ), device='cpu', dtype=torch.float32)
    buf45 = empty((192, ), device='cpu', dtype=torch.float32)
    buf46 = empty_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_6(c_void_p(buf41.data_ptr()), c_void_p(convolution_89.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del convolution_89
    del primals_107
    del squeeze_160
    del unsqueeze_282
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf47 = aten.convolution_backward(buf46, mul_442, primals_242, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_442
    del primals_242
    buf48 = buf47[0]
    buf49 = buf47[1]
    del buf47
    buf50 = reinterpret_tensor(buf25, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf25  # reuse
    buf51 = reinterpret_tensor(buf50, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf50  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_7(c_void_p(buf51.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(add_275.data_ptr()), c_void_p(convolution_88.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf52 = aten.convolution_backward(buf51, mul_441, primals_240, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf51
    del mul_441
    del primals_240
    buf53 = buf52[0]
    buf54 = buf52[1]
    buf55 = buf52[2]
    del buf52
    buf56 = reinterpret_tensor(buf53, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf53  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_8(c_void_p(buf56.data_ptr()), c_void_p(convolution_87.data_ptr()))
    del convolution_87
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf57 = aten.convolution_backward(buf56, mean_17, primals_238, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf56
    del mean_17
    del primals_238
    buf58 = buf57[0]
    buf59 = buf57[1]
    buf60 = buf57[2]
    del buf57
    buf61 = buf37; del buf37  # reuse
    buf62 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf63 = buf48; del buf48  # reuse
    buf64 = buf62; del buf62  # reuse
    buf65 = buf63; del buf63  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_9(c_void_p(buf65.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(convolution_88.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(add_275.data_ptr()), c_void_p(convolution_86.data_ptr()), c_void_p(unsqueeze_294.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf61.data_ptr()))
    del add_275
    del convolution_86
    del convolution_88
    del primals_105
    del squeeze_157
    del unsqueeze_294
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf66 = aten.convolution_backward(buf65, mul_432, primals_237, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf65
    del mul_432
    del primals_237
    buf67 = buf66[0]
    buf68 = buf66[1]
    del buf66
    buf69 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf70 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf71 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf72 = buf67; del buf67  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_10(c_void_p(buf72.data_ptr()), c_void_p(mul_564.data_ptr()), c_void_p(convolution_85.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    del convolution_85
    del mul_564
    del primals_103
    del squeeze_154
    del unsqueeze_306
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf73 = aten.convolution_backward(buf72, add_265, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_265
    del buf72
    del primals_236
    buf74 = buf73[0]
    buf75 = buf73[1]
    del buf73
    buf76 = buf44; del buf44  # reuse
    buf77 = empty((192, ), device='cpu', dtype=torch.float32)
    buf78 = empty((192, ), device='cpu', dtype=torch.float32)
    buf79 = buf46; del buf46  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_11(c_void_p(buf41.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(convolution_84.data_ptr()), c_void_p(unsqueeze_318.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del convolution_84
    del primals_101
    del squeeze_151
    del unsqueeze_318
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf80 = aten.convolution_backward(buf79, mul_417, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_417
    del primals_235
    buf81 = buf80[0]
    buf82 = buf80[1]
    del buf80
    buf83 = reinterpret_tensor(buf58, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf58  # reuse
    buf84 = reinterpret_tensor(buf83, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf83  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_12(c_void_p(buf84.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(add_259.data_ptr()), c_void_p(convolution_83.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf85 = aten.convolution_backward(buf84, mul_416, primals_233, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf84
    del mul_416
    del primals_233
    buf86 = buf85[0]
    buf87 = buf85[1]
    buf88 = buf85[2]
    del buf85
    buf89 = reinterpret_tensor(buf86, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf86  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_13(c_void_p(buf89.data_ptr()), c_void_p(convolution_82.data_ptr()))
    del convolution_82
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf90 = aten.convolution_backward(buf89, mean_16, primals_231, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf89
    del mean_16
    del primals_231
    buf91 = buf90[0]
    buf92 = buf90[1]
    buf93 = buf90[2]
    del buf90
    buf94 = buf70; del buf70  # reuse
    buf95 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf96 = buf81; del buf81  # reuse
    buf97 = buf95; del buf95  # reuse
    buf98 = buf96; del buf96  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14(c_void_p(buf98.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(convolution_83.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(add_259.data_ptr()), c_void_p(convolution_81.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf94.data_ptr()))
    del add_259
    del convolution_81
    del convolution_83
    del primals_99
    del squeeze_148
    del unsqueeze_330
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf99 = aten.convolution_backward(buf98, mul_407, primals_230, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf98
    del mul_407
    del primals_230
    buf100 = buf99[0]
    buf101 = buf99[1]
    del buf99
    buf102 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf103 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf104 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf105 = buf100; del buf100  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_15(c_void_p(buf105.data_ptr()), c_void_p(mul_604.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    del convolution_80
    del mul_604
    del primals_97
    del squeeze_145
    del unsqueeze_342
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf106 = aten.convolution_backward(buf105, add_249, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_249
    del buf105
    del primals_229
    buf107 = buf106[0]
    buf108 = buf106[1]
    del buf106
    buf109 = buf77; del buf77  # reuse
    buf110 = empty((192, ), device='cpu', dtype=torch.float32)
    buf111 = buf79; del buf79  # reuse
    buf112 = buf110; del buf110  # reuse
    cpp_fused_add_native_batch_norm_backward_16(c_void_p(buf112.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()))
    del convolution_79
    del primals_95
    del squeeze_142
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf113 = aten.convolution_backward(buf111, mul_392, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_392
    del primals_228
    buf114 = buf113[0]
    buf115 = buf113[1]
    del buf113
    buf116 = reinterpret_tensor(buf91, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf91  # reuse
    buf117 = reinterpret_tensor(buf116, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf116  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_17(c_void_p(buf117.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(add_243.data_ptr()), c_void_p(convolution_78.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf118 = aten.convolution_backward(buf117, mul_391, primals_226, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf117
    del mul_391
    del primals_226
    buf119 = buf118[0]
    buf120 = buf118[1]
    buf121 = buf118[2]
    del buf118
    buf122 = reinterpret_tensor(buf119, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf119  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_18(c_void_p(buf122.data_ptr()), c_void_p(convolution_77.data_ptr()))
    del convolution_77
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf123 = aten.convolution_backward(buf122, mean_15, primals_224, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf122
    del mean_15
    del primals_224
    buf124 = buf123[0]
    buf125 = buf123[1]
    buf126 = buf123[2]
    del buf123
    buf127 = buf103; del buf103  # reuse
    buf128 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf129 = buf114; del buf114  # reuse
    buf130 = buf128; del buf128  # reuse
    buf131 = buf129; del buf129  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_19(c_void_p(buf131.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(convolution_78.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(add_243.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf127.data_ptr()))
    del add_243
    del convolution_76
    del convolution_78
    del primals_93
    del squeeze_139
    del unsqueeze_366
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf132 = aten.convolution_backward(buf131, mul_382, primals_223, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf131
    del mul_382
    del primals_223
    buf133 = buf132[0]
    buf134 = buf132[1]
    del buf132
    buf135 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf136 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf137 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf138 = buf133; del buf133  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_20(c_void_p(buf138.data_ptr()), c_void_p(mul_644.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()))
    del convolution_75
    del mul_644
    del primals_91
    del squeeze_136
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf139 = aten.convolution_backward(buf138, add_233, primals_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_233
    del buf138
    del primals_222
    buf140 = buf139[0]
    buf141 = buf139[1]
    del buf139
    buf142 = empty((192, ), device='cpu', dtype=torch.float32)
    buf143 = empty((192, ), device='cpu', dtype=torch.float32)
    buf144 = buf111; del buf111  # reuse
    buf146 = buf144; del buf144  # reuse
    buf145 = buf143; del buf143  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_21(c_void_p(buf146.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf142.data_ptr()))
    del convolution_74
    del primals_89
    del squeeze_133
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf147 = aten.convolution_backward(buf146, mul_367, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf146
    del mul_367
    del primals_221
    buf148 = buf147[0]
    buf149 = buf147[1]
    del buf147
    buf150 = reinterpret_tensor(buf124, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf124  # reuse
    buf151 = reinterpret_tensor(buf150, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf150  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_22(c_void_p(buf151.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(add_227.data_ptr()), c_void_p(convolution_73.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf152 = aten.convolution_backward(buf151, mul_366, primals_219, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf151
    del mul_366
    del primals_219
    buf153 = buf152[0]
    buf154 = buf152[1]
    buf155 = buf152[2]
    del buf152
    buf156 = reinterpret_tensor(buf153, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf153  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_23(c_void_p(buf156.data_ptr()), c_void_p(convolution_72.data_ptr()))
    del convolution_72
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf157 = aten.convolution_backward(buf156, mean_14, primals_217, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf156
    del mean_14
    del primals_217
    buf158 = buf157[0]
    buf159 = buf157[1]
    buf160 = buf157[2]
    del buf157
    buf161 = buf136; del buf136  # reuse
    buf162 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf163 = buf148; del buf148  # reuse
    buf164 = buf162; del buf162  # reuse
    buf165 = buf163; del buf163  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_24(c_void_p(buf165.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(add_227.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf161.data_ptr()))
    del add_227
    del buf158
    del convolution_71
    del convolution_73
    del primals_87
    del squeeze_130
    del unsqueeze_402
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf166 = aten.convolution_backward(buf165, mul_357, primals_216, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf165
    del mul_357
    del primals_216
    buf167 = buf166[0]
    buf168 = buf166[1]
    del buf166
    buf169 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf170 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf171 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf172 = buf167; del buf167  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_25(c_void_p(buf172.data_ptr()), c_void_p(mul_684.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    del convolution_70
    del mul_684
    del primals_85
    del squeeze_127
    del unsqueeze_414
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf173 = aten.convolution_backward(buf172, add_217, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_217
    del buf172
    del primals_215
    buf174 = buf173[0]
    buf175 = buf173[1]
    del buf173
    buf176 = buf107; del buf107  # reuse
    buf177 = empty((192, ), device='cpu', dtype=torch.float32)
    buf178 = empty((192, ), device='cpu', dtype=torch.float32)
    buf179 = empty((192, ), device='cpu', dtype=torch.float32)
    buf180 = buf176; del buf176  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_26(c_void_p(buf180.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()))
    del buf140
    del buf174
    del buf178
    del buf41
    del buf74
    del convolution_69
    del primals_83
    del squeeze_124
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf181 = aten.convolution_backward(buf180, mul_342, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf180
    del mul_342
    del primals_214
    buf182 = buf181[0]
    buf183 = buf181[1]
    del buf181
    buf184 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf185 = reinterpret_tensor(buf184, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf184  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_27(c_void_p(buf185.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(add_212.data_ptr()), c_void_p(convolution_68.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf186 = aten.convolution_backward(buf185, mul_341, primals_212, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf185
    del mul_341
    del primals_212
    buf187 = buf186[0]
    buf188 = buf186[1]
    buf189 = buf186[2]
    del buf186
    buf190 = reinterpret_tensor(buf187, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf187  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_28(c_void_p(buf190.data_ptr()), c_void_p(convolution_67.data_ptr()))
    del convolution_67
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf191 = aten.convolution_backward(buf190, mean_13, primals_210, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf190
    del mean_13
    del primals_210
    buf192 = buf191[0]
    buf193 = buf191[1]
    buf194 = buf191[2]
    del buf191
    buf195 = empty((672, ), device='cpu', dtype=torch.float32)
    buf196 = empty((672, ), device='cpu', dtype=torch.float32)
    buf197 = buf182; del buf182  # reuse
    buf198 = buf196; del buf196  # reuse
    buf199 = buf197; del buf197  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_29(c_void_p(buf199.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(add_212.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf195.data_ptr()))
    del add_212
    del convolution_66
    del convolution_68
    del primals_81
    del squeeze_121
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf200 = aten.convolution_backward(buf199, mul_332, primals_209, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf199
    del mul_332
    del primals_209
    buf201 = buf200[0]
    buf202 = buf200[1]
    del buf200
    buf203 = empty((672, ), device='cpu', dtype=torch.float32)
    buf204 = empty((672, ), device='cpu', dtype=torch.float32)
    buf205 = empty((672, ), device='cpu', dtype=torch.float32)
    buf206 = buf201; del buf201  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_30(c_void_p(buf206.data_ptr()), c_void_p(mul_724.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    del convolution_65
    del mul_724
    del primals_79
    del squeeze_118
    del unsqueeze_450
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf207 = aten.convolution_backward(buf206, add_202, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_202
    del buf206
    del primals_208
    buf208 = buf207[0]
    buf209 = buf207[1]
    del buf207
    buf210 = empty((112, ), device='cpu', dtype=torch.float32)
    buf211 = empty((112, ), device='cpu', dtype=torch.float32)
    buf212 = empty((112, ), device='cpu', dtype=torch.float32)
    buf213 = empty_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_31(c_void_p(buf208.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del convolution_64
    del primals_77
    del squeeze_115
    del unsqueeze_462
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf214 = aten.convolution_backward(buf213, mul_317, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_317
    del primals_207
    buf215 = buf214[0]
    buf216 = buf214[1]
    del buf214
    buf217 = reinterpret_tensor(buf192, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf192  # reuse
    buf218 = reinterpret_tensor(buf217, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf217  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_32(c_void_p(buf218.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(add_196.data_ptr()), c_void_p(convolution_63.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf219 = aten.convolution_backward(buf218, mul_316, primals_205, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf218
    del mul_316
    del primals_205
    buf220 = buf219[0]
    buf221 = buf219[1]
    buf222 = buf219[2]
    del buf219
    buf223 = reinterpret_tensor(buf220, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf220  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_33(c_void_p(buf223.data_ptr()), c_void_p(convolution_62.data_ptr()))
    del convolution_62
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf224 = aten.convolution_backward(buf223, mean_12, primals_203, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf223
    del mean_12
    del primals_203
    buf225 = buf224[0]
    buf226 = buf224[1]
    buf227 = buf224[2]
    del buf224
    buf228 = buf204; del buf204  # reuse
    buf229 = empty((672, ), device='cpu', dtype=torch.float32)
    buf230 = buf215; del buf215  # reuse
    buf231 = buf229; del buf229  # reuse
    buf232 = buf230; del buf230  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_34(c_void_p(buf232.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(add_196.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf228.data_ptr()))
    del add_196
    del convolution_61
    del convolution_63
    del primals_75
    del squeeze_112
    del unsqueeze_474
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf233 = aten.convolution_backward(buf232, mul_307, primals_202, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf232
    del mul_307
    del primals_202
    buf234 = buf233[0]
    buf235 = buf233[1]
    del buf233
    buf236 = empty((672, ), device='cpu', dtype=torch.float32)
    buf237 = empty((672, ), device='cpu', dtype=torch.float32)
    buf238 = empty((672, ), device='cpu', dtype=torch.float32)
    buf239 = buf234; del buf234  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_35(c_void_p(buf239.data_ptr()), c_void_p(mul_764.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del convolution_60
    del mul_764
    del primals_73
    del squeeze_109
    del unsqueeze_486
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf240 = aten.convolution_backward(buf239, add_186, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_186
    del buf239
    del primals_201
    buf241 = buf240[0]
    buf242 = buf240[1]
    del buf240
    buf243 = buf211; del buf211  # reuse
    buf244 = empty((112, ), device='cpu', dtype=torch.float32)
    buf245 = empty((112, ), device='cpu', dtype=torch.float32)
    buf246 = buf213; del buf213  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_36(c_void_p(buf208.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()))
    del convolution_59
    del primals_71
    del squeeze_106
    del unsqueeze_498
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf247 = aten.convolution_backward(buf246, mul_292, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_292
    del primals_200
    buf248 = buf247[0]
    buf249 = buf247[1]
    del buf247
    buf250 = reinterpret_tensor(buf225, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf225  # reuse
    buf251 = reinterpret_tensor(buf250, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf250  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_37(c_void_p(buf251.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(add_180.data_ptr()), c_void_p(convolution_58.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf252 = aten.convolution_backward(buf251, mul_291, primals_198, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf251
    del mul_291
    del primals_198
    buf253 = buf252[0]
    buf254 = buf252[1]
    buf255 = buf252[2]
    del buf252
    buf256 = reinterpret_tensor(buf253, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf253  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_38(c_void_p(buf256.data_ptr()), c_void_p(convolution_57.data_ptr()))
    del convolution_57
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf257 = aten.convolution_backward(buf256, mean_11, primals_196, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf256
    del mean_11
    del primals_196
    buf258 = buf257[0]
    buf259 = buf257[1]
    buf260 = buf257[2]
    del buf257
    buf261 = buf237; del buf237  # reuse
    buf262 = empty((672, ), device='cpu', dtype=torch.float32)
    buf263 = buf248; del buf248  # reuse
    buf264 = buf262; del buf262  # reuse
    buf265 = buf263; del buf263  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_39(c_void_p(buf265.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(add_180.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf261.data_ptr()))
    del add_180
    del convolution_56
    del convolution_58
    del primals_69
    del squeeze_103
    del unsqueeze_510
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf266 = aten.convolution_backward(buf265, mul_282, primals_195, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf265
    del mul_282
    del primals_195
    buf267 = buf266[0]
    buf268 = buf266[1]
    del buf266
    buf269 = empty((672, ), device='cpu', dtype=torch.float32)
    buf270 = empty((672, ), device='cpu', dtype=torch.float32)
    buf271 = empty((672, ), device='cpu', dtype=torch.float32)
    buf272 = buf267; del buf267  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_40(c_void_p(buf272.data_ptr()), c_void_p(mul_804.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_522.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    del convolution_55
    del mul_804
    del primals_67
    del squeeze_100
    del unsqueeze_522
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf273 = aten.convolution_backward(buf272, add_170, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_170
    del buf272
    del primals_194
    buf274 = buf273[0]
    buf275 = buf273[1]
    del buf273
    buf276 = buf244; del buf244  # reuse
    buf277 = empty((112, ), device='cpu', dtype=torch.float32)
    buf278 = buf246; del buf246  # reuse
    buf279 = buf277; del buf277  # reuse
    cpp_fused_add_native_batch_norm_backward_41(c_void_p(buf279.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_534.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf278.data_ptr()))
    del convolution_54
    del primals_65
    del squeeze_97
    del unsqueeze_534
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf280 = aten.convolution_backward(buf278, mul_267, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf278
    del mul_267
    del primals_193
    buf281 = buf280[0]
    buf282 = buf280[1]
    del buf280
    buf283 = reinterpret_tensor(buf258, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf258  # reuse
    buf284 = reinterpret_tensor(buf283, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf283  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_42(c_void_p(buf284.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(add_164.data_ptr()), c_void_p(convolution_53.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf285 = aten.convolution_backward(buf284, mul_266, primals_191, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf284
    del mul_266
    del primals_191
    buf286 = buf285[0]
    buf287 = buf285[1]
    buf288 = buf285[2]
    del buf285
    buf289 = reinterpret_tensor(buf286, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf286  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_43(c_void_p(buf289.data_ptr()), c_void_p(convolution_52.data_ptr()))
    del convolution_52
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf290 = aten.convolution_backward(buf289, mean_10, primals_189, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf289
    del mean_10
    del primals_189
    buf291 = buf290[0]
    buf292 = buf290[1]
    buf293 = buf290[2]
    del buf290
    buf294 = buf270; del buf270  # reuse
    buf295 = empty((672, ), device='cpu', dtype=torch.float32)
    buf296 = buf281; del buf281  # reuse
    buf297 = buf295; del buf295  # reuse
    buf298 = buf296; del buf296  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_44(c_void_p(buf298.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(add_164.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf294.data_ptr()))
    del add_164
    del buf291
    del convolution_51
    del convolution_53
    del primals_63
    del squeeze_94
    del unsqueeze_546
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf299 = aten.convolution_backward(buf298, mul_257, primals_188, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf298
    del mul_257
    del primals_188
    buf300 = buf299[0]
    buf301 = buf299[1]
    del buf299
    buf302 = empty((672, ), device='cpu', dtype=torch.float32)
    buf303 = empty((672, ), device='cpu', dtype=torch.float32)
    buf304 = empty((672, ), device='cpu', dtype=torch.float32)
    buf305 = buf300; del buf300  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_45(c_void_p(buf305.data_ptr()), c_void_p(mul_844.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_558.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    del buf303
    del convolution_50
    del mul_844
    del primals_61
    del squeeze_91
    del unsqueeze_558
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf306 = aten.convolution_backward(buf305, add_154, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_154
    del buf305
    del primals_187
    buf307 = buf306[0]
    buf308 = buf306[1]
    del buf306
    buf309 = empty((112, ), device='cpu', dtype=torch.float32)
    buf310 = empty((112, ), device='cpu', dtype=torch.float32)
    buf311 = buf208; del buf208  # reuse
    buf313 = buf311; del buf311  # reuse
    buf312 = buf310; del buf310  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_46(c_void_p(buf313.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_570.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf309.data_ptr()))
    del buf241
    del buf274
    del buf307
    del convolution_49
    del primals_59
    del squeeze_88
    del unsqueeze_570
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf314 = aten.convolution_backward(buf313, mul_242, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf313
    del mul_242
    del primals_186
    buf315 = buf314[0]
    buf316 = buf314[1]
    del buf314
    buf317 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf318 = reinterpret_tensor(buf317, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf317  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_47(c_void_p(buf318.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(add_149.data_ptr()), c_void_p(convolution_48.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf319 = aten.convolution_backward(buf318, mul_241, primals_184, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf318
    del mul_241
    del primals_184
    buf320 = buf319[0]
    buf321 = buf319[1]
    buf322 = buf319[2]
    del buf319
    buf323 = reinterpret_tensor(buf320, (8, 20, 1, 1), (20, 1, 1, 1), 0); del buf320  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_48(c_void_p(buf323.data_ptr()), c_void_p(convolution_47.data_ptr()))
    del convolution_47
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf324 = aten.convolution_backward(buf323, mean_9, primals_182, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf323
    del mean_9
    del primals_182
    buf325 = buf324[0]
    buf326 = buf324[1]
    buf327 = buf324[2]
    del buf324
    buf328 = empty((480, ), device='cpu', dtype=torch.float32)
    buf329 = empty((480, ), device='cpu', dtype=torch.float32)
    buf330 = buf315; del buf315  # reuse
    buf331 = buf329; del buf329  # reuse
    buf332 = buf330; del buf330  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_49(c_void_p(buf332.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(add_149.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf328.data_ptr()))
    del add_149
    del convolution_46
    del convolution_48
    del primals_57
    del squeeze_85
    del unsqueeze_582
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf333 = aten.convolution_backward(buf332, mul_232, primals_181, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf332
    del mul_232
    del primals_181
    buf334 = buf333[0]
    buf335 = buf333[1]
    del buf333
    buf336 = empty((480, ), device='cpu', dtype=torch.float32)
    buf337 = empty((480, ), device='cpu', dtype=torch.float32)
    buf338 = empty((480, ), device='cpu', dtype=torch.float32)
    buf339 = buf334; del buf334  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_50(c_void_p(buf339.data_ptr()), c_void_p(mul_884.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_594.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()))
    del convolution_45
    del mul_884
    del primals_55
    del squeeze_82
    del unsqueeze_594
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf340 = aten.convolution_backward(buf339, add_139, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_139
    del buf339
    del primals_180
    buf341 = buf340[0]
    buf342 = buf340[1]
    del buf340
    buf343 = empty((80, ), device='cpu', dtype=torch.float32)
    buf344 = empty((80, ), device='cpu', dtype=torch.float32)
    buf345 = empty((80, ), device='cpu', dtype=torch.float32)
    buf346 = reinterpret_tensor(buf13, (8, 80, 12, 12), (11520, 1, 960, 80), 0); del buf13  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_51(c_void_p(buf341.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_606.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()))
    del convolution_44
    del primals_53
    del squeeze_79
    del unsqueeze_606
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf347 = aten.convolution_backward(buf346, mul_217, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_217
    del primals_179
    buf348 = buf347[0]
    buf349 = buf347[1]
    del buf347
    buf350 = reinterpret_tensor(buf325, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf325  # reuse
    buf351 = reinterpret_tensor(buf350, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf350  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_52(c_void_p(buf351.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(add_133.data_ptr()), c_void_p(convolution_43.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf352 = aten.convolution_backward(buf351, mul_216, primals_177, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf351
    del mul_216
    del primals_177
    buf353 = buf352[0]
    buf354 = buf352[1]
    buf355 = buf352[2]
    del buf352
    buf356 = reinterpret_tensor(buf353, (8, 20, 1, 1), (20, 1, 1, 1), 0); del buf353  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_53(c_void_p(buf356.data_ptr()), c_void_p(convolution_42.data_ptr()))
    del convolution_42
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf357 = aten.convolution_backward(buf356, mean_8, primals_175, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf356
    del mean_8
    del primals_175
    buf358 = buf357[0]
    buf359 = buf357[1]
    buf360 = buf357[2]
    del buf357
    buf361 = buf337; del buf337  # reuse
    buf362 = empty((480, ), device='cpu', dtype=torch.float32)
    buf363 = buf348; del buf348  # reuse
    buf364 = buf362; del buf362  # reuse
    buf365 = buf363; del buf363  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_54(c_void_p(buf365.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(add_133.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf361.data_ptr()))
    del add_133
    del convolution_41
    del convolution_43
    del primals_51
    del squeeze_76
    del unsqueeze_618
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf366 = aten.convolution_backward(buf365, mul_207, primals_174, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf365
    del mul_207
    del primals_174
    buf367 = buf366[0]
    buf368 = buf366[1]
    del buf366
    buf369 = empty((480, ), device='cpu', dtype=torch.float32)
    buf370 = empty((480, ), device='cpu', dtype=torch.float32)
    buf371 = empty((480, ), device='cpu', dtype=torch.float32)
    buf372 = buf367; del buf367  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_55(c_void_p(buf372.data_ptr()), c_void_p(mul_924.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_630.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    del convolution_40
    del mul_924
    del primals_49
    del squeeze_73
    del unsqueeze_630
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf373 = aten.convolution_backward(buf372, add_123, primals_173, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_123
    del buf372
    del primals_173
    buf374 = buf373[0]
    buf375 = buf373[1]
    del buf373
    buf376 = buf344; del buf344  # reuse
    buf377 = empty((80, ), device='cpu', dtype=torch.float32)
    buf378 = empty((80, ), device='cpu', dtype=torch.float32)
    buf379 = buf346; del buf346  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_56(c_void_p(buf341.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_642.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    del convolution_39
    del primals_47
    del squeeze_70
    del unsqueeze_642
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf380 = aten.convolution_backward(buf379, mul_192, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_192
    del primals_172
    buf381 = buf380[0]
    buf382 = buf380[1]
    del buf380
    buf383 = reinterpret_tensor(buf358, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf358  # reuse
    buf384 = reinterpret_tensor(buf383, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf383  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_57(c_void_p(buf384.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(add_117.data_ptr()), c_void_p(convolution_38.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf385 = aten.convolution_backward(buf384, mul_191, primals_170, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf384
    del mul_191
    del primals_170
    buf386 = buf385[0]
    buf387 = buf385[1]
    buf388 = buf385[2]
    del buf385
    buf389 = reinterpret_tensor(buf386, (8, 20, 1, 1), (20, 1, 1, 1), 0); del buf386  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_58(c_void_p(buf389.data_ptr()), c_void_p(convolution_37.data_ptr()))
    del convolution_37
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf390 = aten.convolution_backward(buf389, mean_7, primals_168, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf389
    del mean_7
    del primals_168
    buf391 = buf390[0]
    buf392 = buf390[1]
    buf393 = buf390[2]
    del buf390
    buf394 = buf370; del buf370  # reuse
    buf395 = empty((480, ), device='cpu', dtype=torch.float32)
    buf396 = buf381; del buf381  # reuse
    buf397 = buf395; del buf395  # reuse
    buf398 = buf396; del buf396  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_59(c_void_p(buf398.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(add_117.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf394.data_ptr()))
    del add_117
    del convolution_36
    del convolution_38
    del primals_45
    del squeeze_67
    del unsqueeze_654
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf399 = aten.convolution_backward(buf398, mul_182, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf398
    del mul_182
    del primals_167
    buf400 = buf399[0]
    buf401 = buf399[1]
    del buf399
    buf402 = empty((480, ), device='cpu', dtype=torch.float32)
    buf403 = empty((480, ), device='cpu', dtype=torch.float32)
    buf404 = empty((480, ), device='cpu', dtype=torch.float32)
    buf405 = buf400; del buf400  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_60(c_void_p(buf405.data_ptr()), c_void_p(mul_964.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_666.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    del convolution_35
    del mul_964
    del primals_43
    del squeeze_64
    del unsqueeze_666
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf406 = aten.convolution_backward(buf405, add_107, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_107
    del buf405
    del primals_166
    buf407 = buf406[0]
    buf408 = buf406[1]
    del buf406
    buf409 = buf377; del buf377  # reuse
    buf410 = empty((80, ), device='cpu', dtype=torch.float32)
    buf411 = buf379; del buf379  # reuse
    buf412 = buf410; del buf410  # reuse
    cpp_fused_add_native_batch_norm_backward_61(c_void_p(buf412.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_678.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf411.data_ptr()))
    del convolution_34
    del primals_41
    del squeeze_61
    del unsqueeze_678
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf413 = aten.convolution_backward(buf411, mul_167, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf411
    del mul_167
    del primals_165
    buf414 = buf413[0]
    buf415 = buf413[1]
    del buf413
    buf416 = reinterpret_tensor(buf391, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf391  # reuse
    buf417 = reinterpret_tensor(buf416, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf416  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_62(c_void_p(buf417.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(add_101.data_ptr()), c_void_p(convolution_33.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf418 = aten.convolution_backward(buf417, mul_166, primals_163, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf417
    del mul_166
    del primals_163
    buf419 = buf418[0]
    buf420 = buf418[1]
    buf421 = buf418[2]
    del buf418
    buf422 = reinterpret_tensor(buf419, (8, 20, 1, 1), (20, 1, 1, 1), 0); del buf419  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_63(c_void_p(buf422.data_ptr()), c_void_p(convolution_32.data_ptr()))
    del convolution_32
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf423 = aten.convolution_backward(buf422, mean_6, primals_161, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf422
    del mean_6
    del primals_161
    buf424 = buf423[0]
    buf425 = buf423[1]
    buf426 = buf423[2]
    del buf423
    buf427 = buf403; del buf403  # reuse
    buf428 = empty((480, ), device='cpu', dtype=torch.float32)
    buf429 = buf414; del buf414  # reuse
    buf430 = buf428; del buf428  # reuse
    buf431 = buf429; del buf429  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64(c_void_p(buf431.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(add_101.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf427.data_ptr()))
    del add_101
    del buf424
    del convolution_31
    del convolution_33
    del primals_39
    del squeeze_58
    del unsqueeze_690
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf432 = aten.convolution_backward(buf431, mul_157, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf431
    del mul_157
    del primals_160
    buf433 = buf432[0]
    buf434 = buf432[1]
    del buf432
    buf435 = empty((480, ), device='cpu', dtype=torch.float32)
    buf436 = empty((480, ), device='cpu', dtype=torch.float32)
    buf437 = empty((480, ), device='cpu', dtype=torch.float32)
    buf438 = buf433; del buf433  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_65(c_void_p(buf438.data_ptr()), c_void_p(mul_1004.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_702.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()))
    del buf436
    del convolution_30
    del mul_1004
    del primals_37
    del squeeze_55
    del unsqueeze_702
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf439 = aten.convolution_backward(buf438, add_91, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_91
    del buf438
    del primals_159
    buf440 = buf439[0]
    buf441 = buf439[1]
    del buf439
    buf442 = empty((80, ), device='cpu', dtype=torch.float32)
    buf443 = empty((80, ), device='cpu', dtype=torch.float32)
    buf444 = buf341; del buf341  # reuse
    buf446 = buf444; del buf444  # reuse
    buf445 = buf443; del buf443  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_66(c_void_p(buf446.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_714.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf442.data_ptr()))
    del buf374
    del buf407
    del buf440
    del convolution_29
    del primals_35
    del squeeze_52
    del unsqueeze_714
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf447 = aten.convolution_backward(buf446, mul_142, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf446
    del mul_142
    del primals_158
    buf448 = buf447[0]
    buf449 = buf447[1]
    del buf447
    buf450 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf451 = reinterpret_tensor(buf450, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf450  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_67(c_void_p(buf451.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(add_86.data_ptr()), c_void_p(convolution_28.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf452 = aten.convolution_backward(buf451, mul_141, primals_156, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf451
    del mul_141
    del primals_156
    buf453 = buf452[0]
    buf454 = buf452[1]
    buf455 = buf452[2]
    del buf452
    buf456 = reinterpret_tensor(buf453, (8, 10, 1, 1), (10, 1, 1, 1), 0); del buf453  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_68(c_void_p(buf456.data_ptr()), c_void_p(convolution_27.data_ptr()))
    del convolution_27
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf457 = aten.convolution_backward(buf456, mean_5, primals_154, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf456
    del mean_5
    del primals_154
    buf458 = buf457[0]
    buf459 = buf457[1]
    buf460 = buf457[2]
    del buf457
    buf461 = empty((240, ), device='cpu', dtype=torch.float32)
    buf462 = empty((240, ), device='cpu', dtype=torch.float32)
    buf463 = buf448; del buf448  # reuse
    buf464 = buf462; del buf462  # reuse
    buf465 = buf463; del buf463  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69(c_void_p(buf465.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(add_86.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf461.data_ptr()))
    del add_86
    del convolution_26
    del convolution_28
    del primals_33
    del squeeze_49
    del unsqueeze_726
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf466 = aten.convolution_backward(buf465, mul_132, primals_153, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf465
    del mul_132
    del primals_153
    buf467 = buf466[0]
    buf468 = buf466[1]
    del buf466
    buf469 = empty((240, ), device='cpu', dtype=torch.float32)
    buf470 = empty((240, ), device='cpu', dtype=torch.float32)
    buf471 = empty((240, ), device='cpu', dtype=torch.float32)
    buf472 = buf467; del buf467  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_70(c_void_p(buf472.data_ptr()), c_void_p(mul_1044.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_738.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    del convolution_25
    del mul_1044
    del primals_31
    del squeeze_46
    del unsqueeze_738
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf473 = aten.convolution_backward(buf472, add_76, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_76
    del buf472
    del primals_152
    buf474 = buf473[0]
    buf475 = buf473[1]
    del buf473
    buf476 = empty((40, ), device='cpu', dtype=torch.float32)
    buf477 = empty((40, ), device='cpu', dtype=torch.float32)
    buf478 = empty((40, ), device='cpu', dtype=torch.float32)
    buf479 = empty_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_71(c_void_p(buf474.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_750.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()))
    del convolution_24
    del primals_29
    del squeeze_43
    del unsqueeze_750
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf480 = aten.convolution_backward(buf479, mul_117, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf479
    del mul_117
    del primals_151
    buf481 = buf480[0]
    buf482 = buf480[1]
    del buf480
    buf483 = reinterpret_tensor(buf458, (8, 240, 1, 1), (240, 1, 1920, 1920), 0); del buf458  # reuse
    buf484 = reinterpret_tensor(buf483, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf483  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_72(c_void_p(buf484.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(add_70.data_ptr()), c_void_p(convolution_23.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf485 = aten.convolution_backward(buf484, mul_116, primals_149, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf484
    del mul_116
    del primals_149
    buf486 = buf485[0]
    buf487 = buf485[1]
    buf488 = buf485[2]
    del buf485
    buf489 = reinterpret_tensor(buf486, (8, 10, 1, 1), (10, 1, 1, 1), 0); del buf486  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_73(c_void_p(buf489.data_ptr()), c_void_p(convolution_22.data_ptr()))
    del convolution_22
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf490 = aten.convolution_backward(buf489, mean_4, primals_147, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf489
    del mean_4
    del primals_147
    buf491 = buf490[0]
    buf492 = buf490[1]
    buf493 = buf490[2]
    del buf490
    buf494 = buf470; del buf470  # reuse
    buf495 = empty((240, ), device='cpu', dtype=torch.float32)
    buf496 = buf481; del buf481  # reuse
    buf497 = buf495; del buf495  # reuse
    buf498 = buf496; del buf496  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_74(c_void_p(buf498.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(add_70.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_762.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf494.data_ptr()))
    del add_70
    del buf491
    del convolution_21
    del convolution_23
    del primals_27
    del squeeze_40
    del unsqueeze_762
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf499 = aten.convolution_backward(buf498, mul_107, primals_146, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf498
    del mul_107
    del primals_146
    buf500 = buf499[0]
    buf501 = buf499[1]
    del buf499
    buf502 = empty((240, ), device='cpu', dtype=torch.float32)
    buf503 = empty((240, ), device='cpu', dtype=torch.float32)
    buf504 = empty((240, ), device='cpu', dtype=torch.float32)
    buf505 = buf500; del buf500  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_75(c_void_p(buf505.data_ptr()), c_void_p(mul_1084.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_774.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()))
    del buf503
    del convolution_20
    del mul_1084
    del primals_25
    del squeeze_37
    del unsqueeze_774
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf506 = aten.convolution_backward(buf505, add_60, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_60
    del buf505
    del primals_145
    buf507 = buf506[0]
    buf508 = buf506[1]
    del buf506
    buf509 = buf477; del buf477  # reuse
    buf510 = empty((40, ), device='cpu', dtype=torch.float32)
    buf511 = empty((40, ), device='cpu', dtype=torch.float32)
    buf512 = buf474; del buf474  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_76(c_void_p(buf512.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_786.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()))
    del buf507
    del buf510
    del convolution_19
    del primals_23
    del squeeze_34
    del unsqueeze_786
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf513 = aten.convolution_backward(buf512, mul_92, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf512
    del mul_92
    del primals_144
    buf514 = buf513[0]
    buf515 = buf513[1]
    del buf513
    buf516 = reinterpret_tensor(buf170, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf170  # reuse
    buf517 = reinterpret_tensor(buf516, (8, 144, 1, 1), (144, 1, 1, 1), 0); del buf516  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_77(c_void_p(buf517.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(add_55.data_ptr()), c_void_p(convolution_18.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf518 = aten.convolution_backward(buf517, mul_91, primals_142, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf517
    del mul_91
    del primals_142
    buf519 = buf518[0]
    buf520 = buf518[1]
    buf521 = buf518[2]
    del buf518
    buf522 = reinterpret_tensor(buf519, (8, 6, 1, 1), (6, 1, 1, 1), 0); del buf519  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_78(c_void_p(buf522.data_ptr()), c_void_p(convolution_17.data_ptr()))
    del convolution_17
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf523 = aten.convolution_backward(buf522, mean_3, primals_140, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf522
    del mean_3
    del primals_140
    buf524 = buf523[0]
    buf525 = buf523[1]
    buf526 = buf523[2]
    del buf523
    buf527 = empty((144, ), device='cpu', dtype=torch.float32)
    buf528 = empty((144, ), device='cpu', dtype=torch.float32)
    buf529 = buf514; del buf514  # reuse
    buf530 = buf528; del buf528  # reuse
    buf531 = buf529; del buf529  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_79(c_void_p(buf531.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(add_55.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_798.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf527.data_ptr()))
    del add_55
    del convolution_16
    del convolution_18
    del primals_21
    del squeeze_31
    del unsqueeze_798
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf532 = aten.convolution_backward(buf531, mul_82, primals_139, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False])
    del buf531
    del mul_82
    del primals_139
    buf533 = buf532[0]
    buf534 = buf532[1]
    del buf532
    buf535 = empty((144, ), device='cpu', dtype=torch.float32)
    buf536 = empty((144, ), device='cpu', dtype=torch.float32)
    buf537 = empty((144, ), device='cpu', dtype=torch.float32)
    buf538 = buf533; del buf533  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_80(c_void_p(buf538.data_ptr()), c_void_p(mul_1124.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_810.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()))
    del convolution_15
    del mul_1124
    del primals_19
    del squeeze_28
    del unsqueeze_810
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf539 = aten.convolution_backward(buf538, add_45, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_45
    del buf538
    del primals_138
    buf540 = buf539[0]
    buf541 = buf539[1]
    del buf539
    buf542 = empty((24, ), device='cpu', dtype=torch.float32)
    buf543 = empty((24, ), device='cpu', dtype=torch.float32)
    buf544 = empty((24, ), device='cpu', dtype=torch.float32)
    buf545 = empty_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_81(c_void_p(buf540.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_822.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()))
    del convolution_14
    del primals_17
    del squeeze_25
    del unsqueeze_822
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf546 = aten.convolution_backward(buf545, mul_67, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf545
    del mul_67
    del primals_137
    buf547 = buf546[0]
    buf548 = buf546[1]
    del buf546
    buf549 = reinterpret_tensor(buf524, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf524  # reuse
    buf550 = reinterpret_tensor(buf549, (8, 144, 1, 1), (144, 1, 1, 1), 0); del buf549  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_82(c_void_p(buf550.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(add_39.data_ptr()), c_void_p(convolution_13.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf551 = aten.convolution_backward(buf550, mul_66, primals_135, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf550
    del mul_66
    del primals_135
    buf552 = buf551[0]
    buf553 = buf551[1]
    buf554 = buf551[2]
    del buf551
    buf555 = reinterpret_tensor(buf552, (8, 6, 1, 1), (6, 1, 1, 1), 0); del buf552  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_83(c_void_p(buf555.data_ptr()), c_void_p(convolution_12.data_ptr()))
    del convolution_12
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf556 = aten.convolution_backward(buf555, mean_2, primals_133, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf555
    del mean_2
    del primals_133
    buf557 = buf556[0]
    buf558 = buf556[1]
    buf559 = buf556[2]
    del buf556
    buf560 = buf536; del buf536  # reuse
    buf561 = empty((144, ), device='cpu', dtype=torch.float32)
    buf562 = buf547; del buf547  # reuse
    buf563 = buf561; del buf561  # reuse
    buf564 = buf562; del buf562  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_84(c_void_p(buf564.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(add_39.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_834.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf560.data_ptr()))
    del add_39
    del buf557
    del convolution_11
    del convolution_13
    del primals_15
    del squeeze_22
    del unsqueeze_834
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf565 = aten.convolution_backward(buf564, mul_57, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
    del buf564
    del mul_57
    del primals_132
    buf566 = buf565[0]
    buf567 = buf565[1]
    del buf565
    buf568 = empty((144, ), device='cpu', dtype=torch.float32)
    buf569 = empty((144, ), device='cpu', dtype=torch.float32)
    buf570 = empty((144, ), device='cpu', dtype=torch.float32)
    buf571 = buf566; del buf566  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_85(c_void_p(buf571.data_ptr()), c_void_p(mul_1164.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_846.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()))
    del buf569
    del convolution_10
    del mul_1164
    del primals_13
    del squeeze_19
    del unsqueeze_846
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf572 = aten.convolution_backward(buf571, add_29, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_29
    del buf571
    del primals_131
    buf573 = buf572[0]
    buf574 = buf572[1]
    del buf572
    buf575 = buf543; del buf543  # reuse
    buf576 = empty((24, ), device='cpu', dtype=torch.float32)
    buf577 = empty((24, ), device='cpu', dtype=torch.float32)
    buf578 = buf540; del buf540  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_86(c_void_p(buf578.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_858.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()))
    del buf573
    del buf576
    del convolution_9
    del primals_11
    del squeeze_16
    del unsqueeze_858
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf579 = aten.convolution_backward(buf578, mul_42, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf578
    del mul_42
    del primals_130
    buf580 = buf579[0]
    buf581 = buf579[1]
    del buf579
    buf582 = empty_strided((8, 96, 1, 1), (96, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf583 = reinterpret_tensor(buf582, (8, 96, 1, 1), (96, 1, 1, 1), 0); del buf582  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_87(c_void_p(buf583.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_8.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf584 = aten.convolution_backward(buf583, mul_41, primals_128, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf583
    del mul_41
    del primals_128
    buf585 = buf584[0]
    buf586 = buf584[1]
    buf587 = buf584[2]
    del buf584
    buf588 = reinterpret_tensor(buf585, (8, 4, 1, 1), (4, 1, 1, 1), 0); del buf585  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_88(c_void_p(buf588.data_ptr()), c_void_p(convolution_7.data_ptr()))
    del convolution_7
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf589 = aten.convolution_backward(buf588, mean_1, primals_126, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_1
    del primals_126
    buf590 = buf589[0]
    buf591 = buf589[1]
    buf592 = buf589[2]
    del buf589
    buf593 = empty((96, ), device='cpu', dtype=torch.float32)
    buf594 = empty((96, ), device='cpu', dtype=torch.float32)
    buf595 = buf580; del buf580  # reuse
    buf596 = buf594; del buf594  # reuse
    buf597 = buf595; del buf595  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_89(c_void_p(buf597.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_870.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf593.data_ptr()))
    del add_24
    del buf590
    del convolution_6
    del convolution_8
    del primals_9
    del squeeze_13
    del unsqueeze_870
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf598 = aten.convolution_backward(buf597, mul_32, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False])
    del buf597
    del mul_32
    del primals_125
    buf599 = buf598[0]
    buf600 = buf598[1]
    del buf598
    buf601 = empty((96, ), device='cpu', dtype=torch.float32)
    buf602 = empty((96, ), device='cpu', dtype=torch.float32)
    buf603 = empty((96, ), device='cpu', dtype=torch.float32)
    buf604 = buf599; del buf599  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_90(c_void_p(buf604.data_ptr()), c_void_p(mul_1204.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_882.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf603.data_ptr()))
    del buf602
    del convolution_5
    del mul_1204
    del primals_7
    del squeeze_10
    del unsqueeze_882
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf605 = aten.convolution_backward(buf604, add_14, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_14
    del buf604
    del primals_124
    buf606 = buf605[0]
    buf607 = buf605[1]
    del buf605
    buf608 = empty((16, ), device='cpu', dtype=torch.float32)
    buf609 = empty((16, ), device='cpu', dtype=torch.float32)
    buf610 = empty((16, ), device='cpu', dtype=torch.float32)
    buf611 = buf606; del buf606  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_91(c_void_p(buf611.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_894.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf610.data_ptr()))
    del buf609
    del convolution_4
    del primals_5
    del squeeze_7
    del unsqueeze_894
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf612 = aten.convolution_backward(buf611, mul_17, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf611
    del mul_17
    del primals_123
    buf613 = buf612[0]
    buf614 = buf612[1]
    del buf612
    buf615 = empty_strided((8, 32, 1, 1), (32, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf616 = reinterpret_tensor(buf615, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf615  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_92(c_void_p(buf616.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(add_9.data_ptr()), c_void_p(convolution_3.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf617 = aten.convolution_backward(buf616, mul_16, primals_121, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf616
    del mul_16
    del primals_121
    buf618 = buf617[0]
    buf619 = buf617[1]
    buf620 = buf617[2]
    del buf617
    buf621 = reinterpret_tensor(buf618, (8, 8, 1, 1), (8, 1, 1, 1), 0); del buf618  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_93(c_void_p(buf621.data_ptr()), c_void_p(convolution_2.data_ptr()))
    del convolution_2
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf622 = aten.convolution_backward(buf621, mean, primals_119, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf621
    del mean
    del primals_119
    buf623 = buf622[0]
    buf624 = buf622[1]
    buf625 = buf622[2]
    del buf622
    buf626 = reinterpret_tensor(buf588, (32, ), (1, ), 0); del buf588  # reuse
    buf627 = empty((32, ), device='cpu', dtype=torch.float32)
    buf628 = buf613; del buf613  # reuse
    buf629 = buf627; del buf627  # reuse
    buf630 = buf628; del buf628  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_94(c_void_p(buf630.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(add_9.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_906.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf626.data_ptr()))
    del add_9
    del buf623
    del convolution_1
    del convolution_3
    del primals_3
    del squeeze_4
    del unsqueeze_906
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf631 = aten.convolution_backward(buf630, mul_7, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf630
    del mul_7
    del primals_118
    buf632 = buf631[0]
    buf633 = buf631[1]
    del buf631
    buf634 = empty((32, ), device='cpu', dtype=torch.float32)
    buf635 = empty((32, ), device='cpu', dtype=torch.float32)
    buf636 = empty((32, ), device='cpu', dtype=torch.float32)
    buf637 = buf632; del buf632  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_95(c_void_p(buf637.data_ptr()), c_void_p(mul_1244.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_918.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(buf636.data_ptr()))
    del buf635
    del convolution
    del mul_1244
    del primals_1
    del squeeze_1
    del unsqueeze_918
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf638 = aten.convolution_backward(buf637, primals_427, primals_117, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf637
    del primals_117
    del primals_427
    buf639 = buf638[1]
    return (buf636, buf634, buf629, buf626, buf610, buf608, buf603, buf601, buf596, buf593, buf577, buf575, buf570, buf568, buf563, buf560, buf544, buf542, buf537, buf535, buf530, buf527, buf511, buf509, buf504, buf502, buf497, buf494, buf478, buf476, buf471, buf469, buf464, buf461, buf445, buf442, buf437, buf435, buf430, buf427, buf412, buf409, buf404, buf402, buf397, buf394, buf378, buf376, buf371, buf369, buf364, buf361, buf345, buf343, buf338, buf336, buf331, buf328, buf312, buf309, buf304, buf302, buf297, buf294, buf279, buf276, buf271, buf269, buf264, buf261, buf245, buf243, buf238, buf236, buf231, buf228, buf212, buf210, buf205, buf203, buf198, buf195, buf179, buf177, buf171, buf169, buf164, buf161, buf145, buf142, buf137, buf135, buf130, buf127, buf112, buf109, buf104, buf102, buf97, buf94, buf78, buf76, buf71, buf69, buf64, buf61, buf45, buf43, buf38, buf36, buf31, buf28, buf12, buf10, buf5, buf3, buf639, buf633, buf624, buf625, buf619, buf620, buf614, buf607, buf600, buf591, buf592, buf586, buf587, buf581, buf574, buf567, buf558, buf559, buf553, buf554, buf548, buf541, buf534, buf525, buf526, buf520, buf521, buf515, buf508, buf501, buf492, buf493, buf487, buf488, buf482, buf475, buf468, buf459, buf460, buf454, buf455, buf449, buf441, buf434, buf425, buf426, buf420, buf421, buf415, buf408, buf401, buf392, buf393, buf387, buf388, buf382, buf375, buf368, buf359, buf360, buf354, buf355, buf349, buf342, buf335, buf326, buf327, buf321, buf322, buf316, buf308, buf301, buf292, buf293, buf287, buf288, buf282, buf275, buf268, buf259, buf260, buf254, buf255, buf249, buf242, buf235, buf226, buf227, buf221, buf222, buf216, buf209, buf202, buf193, buf194, buf188, buf189, buf183, buf175, buf168, buf159, buf160, buf154, buf155, buf149, buf141, buf134, buf125, buf126, buf120, buf121, buf115, buf108, buf101, buf92, buf93, buf87, buf88, buf82, buf75, buf68, buf59, buf60, buf54, buf55, buf49, buf42, buf35, buf26, buf27, buf21, buf22, buf16, buf9, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


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
    primals_23 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_427 = rand_strided((8, 3, 192, 192), (110592, 1, 576, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    mul_7 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_9 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    mul_17 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 16, 96, 96), (147456, 1, 1536, 16), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    add_14 = rand_strided((8, 16, 96, 96), (147456, 1, 1536, 16), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 96, 96, 96), (884736, 1, 9216, 96), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    mul_32 = rand_strided((8, 96, 96, 96), (884736, 1, 9216, 96), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    add_24 = rand_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 4, 1, 1), (4, 1, 4, 4), device='cpu', dtype=torch.float32)
    mul_41 = rand_strided((8, 4, 1, 1), (4, 1, 4, 4), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    mul_42 = rand_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_29 = rand_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    mul_57 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    add_39 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    mul_66 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    mul_67 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_45 = rand_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    mul_82 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    add_55 = rand_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    mul_91 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    mul_92 = rand_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_60 = rand_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    mul_107 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    add_70 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    mul_116 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    mul_117 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_76 = rand_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    mul_132 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    add_86 = rand_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    mul_141 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    mul_142 = rand_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_91 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    mul_157 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_101 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_166 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_167 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_107 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    mul_182 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_117 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_191 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_192 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_123 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    mul_207 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_133 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_216 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_217 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_139 = rand_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    mul_232 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_149 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_241 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_242 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_154 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    mul_257 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    add_164 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_266 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_267 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_170 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    mul_282 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    add_180 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_291 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_292 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_186 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    mul_307 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    add_196 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    mean_12 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_316 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_317 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_202 = rand_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    mul_332 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    add_212 = rand_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cpu', dtype=torch.float32)
    mean_13 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_341 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_342 = rand_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_217 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    mul_357 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    add_227 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    mean_14 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_366 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_367 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_233 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    mul_382 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    add_243 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    mean_15 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_391 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_78 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_392 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_249 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    mul_407 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_81 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    add_259 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    mean_16 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_82 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_416 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_83 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_417 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_84 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_265 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    convolution_85 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    mul_432 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_86 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    add_275 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    mean_17 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_87 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_441 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_88 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_442 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_89 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_281 = rand_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cpu', dtype=torch.float32)
    convolution_90 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    mul_457 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_91 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    add_291 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    mean_18 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_92 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_466 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_93 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_467 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    convolution_94 = rand_strided((8, 320, 6, 6), (11520, 1, 1920, 320), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    add_296 = rand_strided((8, 320, 6, 6), (11520, 1, 1920, 320), device='cpu', dtype=torch.float32)
    convolution_95 = rand_strided((8, 1280, 6, 6), (46080, 1, 7680, 1280), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    view = rand_strided((8, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    mul_484 = rand_strided((8, 1280, 6, 6), (46080, 1, 7680, 1280), device='cpu', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_524 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_564 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_604 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_644 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_684 = rand_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cpu', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_724 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_764 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_804 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_844 = rand_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cpu', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_884 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_924 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_964 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1004 = rand_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cpu', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1044 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cpu', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1084 = rand_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cpu', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_786 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_798 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1124 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cpu', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_822 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1164 = rand_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cpu', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_858 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_870 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1204 = rand_strided((8, 96, 96, 96), (884736, 1, 9216, 96), device='cpu', dtype=torch.float32)
    unsqueeze_882 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_894 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_906 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1244 = rand_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cpu', dtype=torch.float32)
    unsqueeze_918 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_118, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_128, primals_130, primals_131, primals_132, primals_133, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_144, primals_145, primals_146, primals_147, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_161, primals_163, primals_165, primals_166, primals_167, primals_168, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_187, primals_188, primals_189, primals_191, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_233, primals_235, primals_236, primals_237, primals_238, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_249, primals_250, primals_427, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, add_9, mean, convolution_2, mul_16, convolution_3, mul_17, convolution_4, squeeze_7, add_14, convolution_5, squeeze_10, mul_32, convolution_6, squeeze_13, add_24, mean_1, convolution_7, mul_41, convolution_8, mul_42, convolution_9, squeeze_16, add_29, convolution_10, squeeze_19, mul_57, convolution_11, squeeze_22, add_39, mean_2, convolution_12, mul_66, convolution_13, mul_67, convolution_14, squeeze_25, add_45, convolution_15, squeeze_28, mul_82, convolution_16, squeeze_31, add_55, mean_3, convolution_17, mul_91, convolution_18, mul_92, convolution_19, squeeze_34, add_60, convolution_20, squeeze_37, mul_107, convolution_21, squeeze_40, add_70, mean_4, convolution_22, mul_116, convolution_23, mul_117, convolution_24, squeeze_43, add_76, convolution_25, squeeze_46, mul_132, convolution_26, squeeze_49, add_86, mean_5, convolution_27, mul_141, convolution_28, mul_142, convolution_29, squeeze_52, add_91, convolution_30, squeeze_55, mul_157, convolution_31, squeeze_58, add_101, mean_6, convolution_32, mul_166, convolution_33, mul_167, convolution_34, squeeze_61, add_107, convolution_35, squeeze_64, mul_182, convolution_36, squeeze_67, add_117, mean_7, convolution_37, mul_191, convolution_38, mul_192, convolution_39, squeeze_70, add_123, convolution_40, squeeze_73, mul_207, convolution_41, squeeze_76, add_133, mean_8, convolution_42, mul_216, convolution_43, mul_217, convolution_44, squeeze_79, add_139, convolution_45, squeeze_82, mul_232, convolution_46, squeeze_85, add_149, mean_9, convolution_47, mul_241, convolution_48, mul_242, convolution_49, squeeze_88, add_154, convolution_50, squeeze_91, mul_257, convolution_51, squeeze_94, add_164, mean_10, convolution_52, mul_266, convolution_53, mul_267, convolution_54, squeeze_97, add_170, convolution_55, squeeze_100, mul_282, convolution_56, squeeze_103, add_180, mean_11, convolution_57, mul_291, convolution_58, mul_292, convolution_59, squeeze_106, add_186, convolution_60, squeeze_109, mul_307, convolution_61, squeeze_112, add_196, mean_12, convolution_62, mul_316, convolution_63, mul_317, convolution_64, squeeze_115, add_202, convolution_65, squeeze_118, mul_332, convolution_66, squeeze_121, add_212, mean_13, convolution_67, mul_341, convolution_68, mul_342, convolution_69, squeeze_124, add_217, convolution_70, squeeze_127, mul_357, convolution_71, squeeze_130, add_227, mean_14, convolution_72, mul_366, convolution_73, mul_367, convolution_74, squeeze_133, add_233, convolution_75, squeeze_136, mul_382, convolution_76, squeeze_139, add_243, mean_15, convolution_77, mul_391, convolution_78, mul_392, convolution_79, squeeze_142, add_249, convolution_80, squeeze_145, mul_407, convolution_81, squeeze_148, add_259, mean_16, convolution_82, mul_416, convolution_83, mul_417, convolution_84, squeeze_151, add_265, convolution_85, squeeze_154, mul_432, convolution_86, squeeze_157, add_275, mean_17, convolution_87, mul_441, convolution_88, mul_442, convolution_89, squeeze_160, add_281, convolution_90, squeeze_163, mul_457, convolution_91, squeeze_166, add_291, mean_18, convolution_92, mul_466, convolution_93, mul_467, convolution_94, squeeze_169, add_296, convolution_95, squeeze_172, view, permute_1, mul_484, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_524, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_564, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_604, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_644, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_684, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_724, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_764, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_804, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_844, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_884, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_924, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_964, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_1004, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1044, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1084, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1124, unsqueeze_810, unsqueeze_822, unsqueeze_834, mul_1164, unsqueeze_846, unsqueeze_858, unsqueeze_870, mul_1204, unsqueeze_882, unsqueeze_894, unsqueeze_906, mul_1244, unsqueeze_918, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tinynet_a', benchmark_compiled_module)
