
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1280L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1280L*x2) + (62720L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1280L*x2) + (62720L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(49.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(8L))
                    {
                        float tmp24[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1280L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (62720L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (62720L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(0.002551020408163265);
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
                        at::vec::transpose_mxn<float,8,8>(tmp24, 8, out_ptr4 + static_cast<long>(x1 + (49L*x2) + (62720L*x0)), static_cast<long>(49L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (1280L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (1280L*x1) + (62720L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (1280L*x1) + (62720L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(x2)];
                        auto tmp8 = out_ptr2[static_cast<long>(x2)];
                        auto tmp11 = in_ptr5[static_cast<long>(x2)];
                        auto tmp16 = out_ptr1[static_cast<long>(x2)];
                        auto tmp19 = in_ptr6[static_cast<long>(x2)];
                        auto tmp1 = static_cast<float>(49.0);
                        auto tmp2 = tmp0 / tmp1;
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp9 = static_cast<float>(0.002551020408163265);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp12 = decltype(tmp11)(tmp11 * tmp11);
                        auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                        auto tmp14 = decltype(tmp7)(tmp7 * tmp13);
                        auto tmp15 = decltype(tmp4)(tmp4 - tmp14);
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp9);
                        auto tmp18 = decltype(tmp15)(tmp15 - tmp17);
                        auto tmp20 = decltype(tmp11)(tmp11 * tmp19);
                        auto tmp21 = decltype(tmp18)(tmp18 * tmp20);
                        out_ptr4[static_cast<long>(x1 + (49L*x2) + (62720L*x0))] = tmp21;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x2) + (56448L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x2) + (56448L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x2) + (56448L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x2) + (56448L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x2) + (56448L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x2) + (56448L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x2) + (56448L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x2) + (56448L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1152L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1152L*x2) + (56448L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x2) + (32928L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x2) + (32928L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_23 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x2) + (32928L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x2) + (32928L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (672L*x2) + (32928L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
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


cpp_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_25 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x3) + (9408L*x2) + (131712L*x1)));
                                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x3) + (9408L*x2) + (131712L*x1)));
                                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp0 = c10::convert<int>(1L + x2);
                                auto tmp1 = static_cast<int>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<int>(17);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = c10::convert<int>(1L + x3);
                                auto tmp6 = tmp5 >= tmp1;
                                auto tmp7 = tmp5 < tmp3;
                                auto tmp8 = tmp2 & tmp4;
                                auto tmp9 = tmp8 & tmp6;
                                auto tmp10 = tmp9 & tmp7;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(in_ptr0 + static_cast<long>(12096L + x0 + (672L*x3) + (11424L*x2) + (194208L*x1)), to_float_mask(tmp10));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                                auto tmp15 = tmp13 * tmp14;
                                auto tmp16 = [&]
                                {
                                    auto tmp17 = masked_load(in_ptr0 + static_cast<long>(12096L + x0 + (672L*x3) + (11424L*x2) + (194208L*x1)), to_float_mask(tmp10));
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp10));
                                auto tmp19 = tmp18 * tmp14;
                                auto tmp22 = tmp20 - tmp21;
                                auto tmp23 = tmp19 * tmp22;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                                tmp_acc1_vec = tmp_acc1_vec + tmp23;
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(672L); x3+=static_cast<long>(8L))
                        {
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (672L*x2) + (9408L*x1) + (131712L*x0)));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (672L*x2) + (9408L*x1) + (131712L*x0)));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
                            auto tmp28 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3));
                            auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3));
                            auto tmp0 = c10::convert<int>(1L + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(17);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>(1L + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(12096L + x3 + (672L*x2) + (11424L*x1) + (194208L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp18 = tmp16 - tmp17;
                            auto tmp20 = static_cast<float>(0.0006377551020408163);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp19 * tmp21;
                            auto tmp24 = tmp23 * tmp23;
                            auto tmp25 = tmp22 * tmp24;
                            auto tmp26 = tmp18 * tmp25;
                            auto tmp27 = tmp15 - tmp26;
                            auto tmp29 = tmp28 * tmp21;
                            auto tmp30 = tmp27 - tmp29;
                            auto tmp32 = tmp23 * tmp31;
                            auto tmp33 = tmp30 * tmp32;
                            tmp33.store(out_ptr3 + static_cast<long>(x3 + (672L*x2) + (9408L*x1) + (131712L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_26 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (112L*x0)));
                }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x2) + (131712L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x2) + (131712L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (672L*x2) + (131712L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_31 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (112L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x2) + (131712L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x2) + (131712L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (672L*x2) + (131712L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_38 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_43 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_53 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x2) + (47040L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (240L*x2) + (47040L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (240L*x2) + (47040L*x1)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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


cpp_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_55 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                            {
                                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x3) + (6720L*x2) + (188160L*x1)));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x3) + (6720L*x2) + (188160L*x1)));
                                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(29);
                                auto tmp2 = tmp0 < tmp1;
                                auto tmp3 = c10::convert<int>(x3);
                                auto tmp4 = tmp3 < tmp1;
                                auto tmp5 = tmp2 & tmp4;
                                auto tmp6 = [&]
                                {
                                    auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x0 + (240L*x3) + (6960L*x2) + (201840L*x1)), to_float_mask(tmp5));
                                    return tmp7;
                                }
                                ;
                                auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                                auto tmp10 = tmp8 * tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x0 + (240L*x3) + (6960L*x2) + (201840L*x1)), to_float_mask(tmp5));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp5));
                                auto tmp14 = tmp13 * tmp9;
                                auto tmp17 = tmp15 - tmp16;
                                auto tmp18 = tmp14 * tmp17;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp18;
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(8L))
                        {
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (240L*x2) + (6720L*x1) + (188160L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (240L*x2) + (6720L*x1) + (188160L*x0)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3));
                            auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(29);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x3 + (240L*x2) + (6960L*x1) + (201840L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp10 = tmp8 * tmp9;
                            auto tmp13 = tmp11 - tmp12;
                            auto tmp15 = static_cast<float>(0.00015943877551020407);
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 * tmp16;
                            auto tmp19 = tmp18 * tmp18;
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp21 = tmp13 * tmp20;
                            auto tmp22 = tmp10 - tmp21;
                            auto tmp24 = tmp23 * tmp16;
                            auto tmp25 = tmp22 - tmp24;
                            auto tmp27 = tmp18 * tmp26;
                            auto tmp28 = tmp25 * tmp27;
                            tmp28.store(out_ptr3 + static_cast<long>(x3 + (240L*x2) + (6720L*x1) + (188160L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_56 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
                }
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_61 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (144L*x2) + (112896L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x2) + (112896L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_63 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x2) + (112896L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (144L*x2) + (112896L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (144L*x2) + (112896L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (112896L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (112896L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (144L*x1) + (112896L*x0)));
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
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (112896L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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


cpp_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_65 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                            {
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x3) + (8064L*x2) + (451584L*x1)));
                                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x3) + (8064L*x2) + (451584L*x1)));
                                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp0 = c10::convert<int>(1L + x2);
                                auto tmp1 = static_cast<int>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<int>(59);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = c10::convert<int>(1L + x3);
                                auto tmp6 = tmp5 >= tmp1;
                                auto tmp7 = tmp5 < tmp3;
                                auto tmp8 = tmp2 & tmp4;
                                auto tmp9 = tmp8 & tmp6;
                                auto tmp10 = tmp9 & tmp7;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(in_ptr0 + static_cast<long>(8640L + x0 + (144L*x3) + (8496L*x2) + (501264L*x1)), to_float_mask(tmp10));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                                auto tmp15 = tmp13 * tmp14;
                                auto tmp16 = [&]
                                {
                                    auto tmp17 = masked_load(in_ptr0 + static_cast<long>(8640L + x0 + (144L*x3) + (8496L*x2) + (501264L*x1)), to_float_mask(tmp10));
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp10));
                                auto tmp19 = tmp18 * tmp14;
                                auto tmp22 = tmp20 - tmp21;
                                auto tmp23 = tmp19 * tmp22;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                                tmp_acc1_vec = tmp_acc1_vec + tmp23;
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(8L))
                        {
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (144L*x2) + (8064L*x1) + (451584L*x0)));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (144L*x2) + (8064L*x1) + (451584L*x0)));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
                            auto tmp28 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3));
                            auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3));
                            auto tmp0 = c10::convert<int>(1L + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(59);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>(1L + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(8640L + x3 + (144L*x2) + (8496L*x1) + (501264L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp18 = tmp16 - tmp17;
                            auto tmp20 = static_cast<float>(3.985969387755102e-05);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp19 * tmp21;
                            auto tmp24 = tmp23 * tmp23;
                            auto tmp25 = tmp22 * tmp24;
                            auto tmp26 = tmp18 * tmp25;
                            auto tmp27 = tmp15 - tmp26;
                            auto tmp29 = tmp28 * tmp21;
                            auto tmp30 = tmp27 - tmp29;
                            auto tmp32 = tmp23 * tmp31;
                            auto tmp33 = tmp30 * tmp32;
                            tmp33.store(out_ptr3 + static_cast<long>(x3 + (144L*x2) + (8064L*x1) + (451584L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_66 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (144L*x2) + (451584L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x2) + (451584L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_68 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (144L*x2) + (451584L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (144L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (144L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (144L*x2) + (451584L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (144L*x2) + (451584L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(3136.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (451584L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (451584L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (144L*x1) + (451584L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(3136.0);
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
                        auto tmp22 = static_cast<float>(3.985969387755102e-05);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (144L*x1) + (451584L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_71 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x2) + (301056L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x2) + (301056L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_73 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x2) + (301056L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (96L*x2) + (301056L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (96L*x2) + (301056L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(3136.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (96L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (96L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (96L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (96L*x1) + (301056L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (96L*x1) + (301056L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(3136.0);
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
                        auto tmp22 = static_cast<float>(3.985969387755102e-05);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (96L*x1) + (301056L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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


cpp_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_75 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(112L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(112L); x3+=static_cast<long>(1L))
                            {
                                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x3) + (10752L*x2) + (1204224L*x1)));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x3) + (10752L*x2) + (1204224L*x1)));
                                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(113);
                                auto tmp2 = tmp0 < tmp1;
                                auto tmp3 = c10::convert<int>(x3);
                                auto tmp4 = tmp3 < tmp1;
                                auto tmp5 = tmp2 & tmp4;
                                auto tmp6 = [&]
                                {
                                    auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x0 + (96L*x3) + (10848L*x2) + (1225824L*x1)), to_float_mask(tmp5));
                                    return tmp7;
                                }
                                ;
                                auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                                auto tmp10 = tmp8 * tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x0 + (96L*x3) + (10848L*x2) + (1225824L*x1)), to_float_mask(tmp5));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp5));
                                auto tmp14 = tmp13 * tmp9;
                                auto tmp17 = tmp15 - tmp16;
                                auto tmp18 = tmp14 * tmp17;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                                tmp_acc1_vec = tmp_acc1_vec + tmp18;
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(112L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(96L); x3+=static_cast<long>(8L))
                        {
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (96L*x2) + (10752L*x1) + (1204224L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (96L*x2) + (10752L*x1) + (1204224L*x0)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3));
                            auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(113);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x3 + (96L*x2) + (10848L*x1) + (1225824L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp10 = tmp8 * tmp9;
                            auto tmp13 = tmp11 - tmp12;
                            auto tmp15 = static_cast<float>(9.964923469387754e-06);
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 * tmp16;
                            auto tmp19 = tmp18 * tmp18;
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp21 = tmp13 * tmp20;
                            auto tmp22 = tmp10 - tmp21;
                            auto tmp24 = tmp23 * tmp16;
                            auto tmp25 = tmp22 - tmp24;
                            auto tmp27 = tmp18 * tmp26;
                            auto tmp28 = tmp25 * tmp27;
                            tmp28.store(out_ptr3 + static_cast<long>(x3 + (96L*x2) + (10752L*x1) + (1204224L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_76 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x2) + (401408L*x0)));
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_78 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x2) + (401408L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (32L*x2) + (401408L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (32L*x2) + (401408L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(12544.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (32L*x1) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (32L*x1) + (401408L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (32L*x1) + (401408L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(12544.0);
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
                        auto tmp22 = static_cast<float>(9.964923469387754e-06);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (32L*x1) + (401408L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(9.964923469387754e-06);
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
    primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_120, primals_122, primals_123, primals_124, primals_126, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, constant_pad_nd, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, add_9, mean, convolution_2, mul_16, convolution_3, mul_17, convolution_4, squeeze_7, add_14, convolution_5, squeeze_10, constant_pad_nd_1, convolution_6, squeeze_13, add_24, mean_1, convolution_7, mul_41, convolution_8, mul_42, convolution_9, squeeze_16, add_29, convolution_10, squeeze_19, mul_57, convolution_11, squeeze_22, add_39, mean_2, convolution_12, mul_66, convolution_13, mul_67, convolution_14, squeeze_25, add_45, convolution_15, squeeze_28, constant_pad_nd_2, convolution_16, squeeze_31, add_55, mean_3, convolution_17, mul_91, convolution_18, mul_92, convolution_19, squeeze_34, add_60, convolution_20, squeeze_37, mul_107, convolution_21, squeeze_40, add_70, mean_4, convolution_22, mul_116, convolution_23, mul_117, convolution_24, squeeze_43, add_76, convolution_25, squeeze_46, constant_pad_nd_3, convolution_26, squeeze_49, add_86, mean_5, convolution_27, mul_141, convolution_28, mul_142, convolution_29, squeeze_52, add_91, convolution_30, squeeze_55, mul_157, convolution_31, squeeze_58, add_101, mean_6, convolution_32, mul_166, convolution_33, mul_167, convolution_34, squeeze_61, add_107, convolution_35, squeeze_64, mul_182, convolution_36, squeeze_67, add_117, mean_7, convolution_37, mul_191, convolution_38, mul_192, convolution_39, squeeze_70, add_123, convolution_40, squeeze_73, mul_207, convolution_41, squeeze_76, add_133, mean_8, convolution_42, mul_216, convolution_43, mul_217, convolution_44, squeeze_79, add_138, convolution_45, squeeze_82, mul_232, convolution_46, squeeze_85, add_148, mean_9, convolution_47, mul_241, convolution_48, mul_242, convolution_49, squeeze_88, add_154, convolution_50, squeeze_91, mul_257, convolution_51, squeeze_94, add_164, mean_10, convolution_52, mul_266, convolution_53, mul_267, convolution_54, squeeze_97, add_170, convolution_55, squeeze_100, constant_pad_nd_4, convolution_56, squeeze_103, add_180, mean_11, convolution_57, mul_291, convolution_58, mul_292, convolution_59, squeeze_106, add_185, convolution_60, squeeze_109, mul_307, convolution_61, squeeze_112, add_195, mean_12, convolution_62, mul_316, convolution_63, mul_317, convolution_64, squeeze_115, add_201, convolution_65, squeeze_118, mul_332, convolution_66, squeeze_121, add_211, mean_13, convolution_67, mul_341, convolution_68, mul_342, convolution_69, squeeze_124, add_217, convolution_70, squeeze_127, mul_357, convolution_71, squeeze_130, add_227, mean_14, convolution_72, mul_366, convolution_73, mul_367, convolution_74, squeeze_133, add_233, convolution_75, squeeze_136, mul_382, convolution_76, squeeze_139, add_243, mean_15, convolution_77, mul_391, convolution_78, mul_392, convolution_79, squeeze_142, add_248, convolution_80, squeeze_145, view, permute_1, mul_409, unsqueeze_198, unsqueeze_210, unsqueeze_222, mul_449, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_489, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_529, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_569, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_609, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_649, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_689, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_729, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_769, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_809, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_849, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_889, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_929, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_969, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1009, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1049, unsqueeze_774, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_8, (96, ), (1, ))
    assert_size_stride(primals_10, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_15, (144, ), (1, ))
    assert_size_stride(primals_17, (144, ), (1, ))
    assert_size_stride(primals_19, (24, ), (1, ))
    assert_size_stride(primals_21, (144, ), (1, ))
    assert_size_stride(primals_23, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_24, (144, ), (1, ))
    assert_size_stride(primals_26, (40, ), (1, ))
    assert_size_stride(primals_28, (240, ), (1, ))
    assert_size_stride(primals_30, (240, ), (1, ))
    assert_size_stride(primals_32, (40, ), (1, ))
    assert_size_stride(primals_34, (240, ), (1, ))
    assert_size_stride(primals_36, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_37, (240, ), (1, ))
    assert_size_stride(primals_39, (80, ), (1, ))
    assert_size_stride(primals_41, (480, ), (1, ))
    assert_size_stride(primals_43, (480, ), (1, ))
    assert_size_stride(primals_45, (80, ), (1, ))
    assert_size_stride(primals_47, (480, ), (1, ))
    assert_size_stride(primals_49, (480, ), (1, ))
    assert_size_stride(primals_51, (80, ), (1, ))
    assert_size_stride(primals_53, (480, ), (1, ))
    assert_size_stride(primals_55, (480, ), (1, ))
    assert_size_stride(primals_57, (112, ), (1, ))
    assert_size_stride(primals_59, (672, ), (1, ))
    assert_size_stride(primals_61, (672, ), (1, ))
    assert_size_stride(primals_63, (112, ), (1, ))
    assert_size_stride(primals_65, (672, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_69, (112, ), (1, ))
    assert_size_stride(primals_71, (672, ), (1, ))
    assert_size_stride(primals_73, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_74, (672, ), (1, ))
    assert_size_stride(primals_76, (192, ), (1, ))
    assert_size_stride(primals_78, (1152, ), (1, ))
    assert_size_stride(primals_80, (1152, ), (1, ))
    assert_size_stride(primals_82, (192, ), (1, ))
    assert_size_stride(primals_84, (1152, ), (1, ))
    assert_size_stride(primals_86, (1152, ), (1, ))
    assert_size_stride(primals_88, (192, ), (1, ))
    assert_size_stride(primals_90, (1152, ), (1, ))
    assert_size_stride(primals_92, (1152, ), (1, ))
    assert_size_stride(primals_94, (192, ), (1, ))
    assert_size_stride(primals_96, (1152, ), (1, ))
    assert_size_stride(primals_98, (1152, ), (1, ))
    assert_size_stride(primals_100, (320, ), (1, ))
    assert_size_stride(primals_102, (1280, ), (1, ))
    assert_size_stride(primals_104, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_107, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_109, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_110, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_111, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_113, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_115, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_116, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_117, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_118, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_120, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_122, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_123, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_124, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_126, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_128, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_129, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_130, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_131, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_133, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_135, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_136, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_137, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_139, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_141, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_142, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_143, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_144, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_146, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_148, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_149, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_150, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_151, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_153, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_155, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_156, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_157, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_158, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_160, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_162, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_163, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_164, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_165, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_167, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_169, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_170, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_171, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_172, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_174, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_176, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_177, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_178, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_180, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_182, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_183, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_184, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_185, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_187, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_189, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_190, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_191, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_192, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_194, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_196, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_197, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_198, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_199, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_201, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_203, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_204, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_205, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_206, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_208, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_210, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_211, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(constant_pad_nd, (8, 3, 225, 225), (151875, 1, 675, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(mul_7, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(add_9, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(mean, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(convolution_2, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(mul_16, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_3, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(mul_17, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_4, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_14, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_5, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(constant_pad_nd_1, (8, 96, 113, 113), (1225824, 1, 10848, 96))
    assert_size_stride(convolution_6, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(add_24, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(mean_1, (8, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(convolution_7, (8, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(mul_41, (8, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(convolution_8, (8, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(mul_42, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_9, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_16, (24, ), (1, ))
    assert_size_stride(add_29, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_10, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_19, (144, ), (1, ))
    assert_size_stride(mul_57, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_11, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_22, (144, ), (1, ))
    assert_size_stride(add_39, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(mean_2, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_12, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_66, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_13, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_67, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_14, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_25, (24, ), (1, ))
    assert_size_stride(add_45, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_15, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(squeeze_28, (144, ), (1, ))
    assert_size_stride(constant_pad_nd_2, (8, 144, 59, 59), (501264, 1, 8496, 144))
    assert_size_stride(convolution_16, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(squeeze_31, (144, ), (1, ))
    assert_size_stride(add_55, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(mean_3, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_17, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_91, (8, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_18, (8, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_92, (8, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(convolution_19, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_34, (40, ), (1, ))
    assert_size_stride(add_60, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_20, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_37, (240, ), (1, ))
    assert_size_stride(mul_107, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_21, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_40, (240, ), (1, ))
    assert_size_stride(add_70, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(mean_4, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_22, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_116, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_23, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_117, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_24, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_43, (40, ), (1, ))
    assert_size_stride(add_76, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_25, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_46, (240, ), (1, ))
    assert_size_stride(constant_pad_nd_3, (8, 240, 29, 29), (201840, 1, 6960, 240))
    assert_size_stride(convolution_26, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(squeeze_49, (240, ), (1, ))
    assert_size_stride(add_86, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(mean_5, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_27, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_141, (8, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_28, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_142, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_29, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_52, (80, ), (1, ))
    assert_size_stride(add_91, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_30, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_55, (480, ), (1, ))
    assert_size_stride(mul_157, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_31, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_58, (480, ), (1, ))
    assert_size_stride(add_101, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_6, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_32, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_166, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_33, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_167, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_34, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_61, (80, ), (1, ))
    assert_size_stride(add_107, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_35, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_64, (480, ), (1, ))
    assert_size_stride(mul_182, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_36, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_67, (480, ), (1, ))
    assert_size_stride(add_117, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_7, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_37, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_191, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_38, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_192, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_39, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_70, (80, ), (1, ))
    assert_size_stride(add_123, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_40, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_73, (480, ), (1, ))
    assert_size_stride(mul_207, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_41, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_76, (480, ), (1, ))
    assert_size_stride(add_133, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_8, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_42, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_216, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_43, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_217, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_44, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_79, (112, ), (1, ))
    assert_size_stride(add_138, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_45, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_82, (672, ), (1, ))
    assert_size_stride(mul_232, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_46, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_85, (672, ), (1, ))
    assert_size_stride(add_148, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_9, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_47, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_241, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_48, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_242, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_49, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_88, (112, ), (1, ))
    assert_size_stride(add_154, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_50, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_91, (672, ), (1, ))
    assert_size_stride(mul_257, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_51, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_94, (672, ), (1, ))
    assert_size_stride(add_164, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_10, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_52, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_266, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_53, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_267, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_54, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_97, (112, ), (1, ))
    assert_size_stride(add_170, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_55, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(squeeze_100, (672, ), (1, ))
    assert_size_stride(constant_pad_nd_4, (8, 672, 17, 17), (194208, 1, 11424, 672))
    assert_size_stride(convolution_56, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(squeeze_103, (672, ), (1, ))
    assert_size_stride(add_180, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(mean_11, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_57, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_291, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_58, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_292, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(convolution_59, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(squeeze_106, (192, ), (1, ))
    assert_size_stride(add_185, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_60, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_109, (1152, ), (1, ))
    assert_size_stride(mul_307, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_61, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_112, (1152, ), (1, ))
    assert_size_stride(add_195, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_12, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_62, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_316, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_63, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_317, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_64, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(squeeze_115, (192, ), (1, ))
    assert_size_stride(add_201, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_65, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_118, (1152, ), (1, ))
    assert_size_stride(mul_332, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_66, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_121, (1152, ), (1, ))
    assert_size_stride(add_211, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_13, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_67, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_341, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_68, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_342, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_69, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(squeeze_124, (192, ), (1, ))
    assert_size_stride(add_217, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_70, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_127, (1152, ), (1, ))
    assert_size_stride(mul_357, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_71, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_130, (1152, ), (1, ))
    assert_size_stride(add_227, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_14, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_72, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_366, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_73, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_367, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_74, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(squeeze_133, (192, ), (1, ))
    assert_size_stride(add_233, (8, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_75, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_136, (1152, ), (1, ))
    assert_size_stride(mul_382, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_76, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(squeeze_139, (1152, ), (1, ))
    assert_size_stride(add_243, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_15, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_77, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_391, (8, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_78, (8, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_392, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_79, (8, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(squeeze_142, (320, ), (1, ))
    assert_size_stride(add_248, (8, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(convolution_80, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(squeeze_145, (1280, ), (1, ))
    assert_size_stride(view, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(mul_409, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(unsqueeze_198, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_210, (1, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(unsqueeze_222, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_449, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(unsqueeze_234, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_246, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_489, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(unsqueeze_270, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_529, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(unsqueeze_306, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_318, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(mul_569, (8, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(unsqueeze_342, (1, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_609, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(unsqueeze_378, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_649, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(unsqueeze_414, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(mul_689, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(unsqueeze_450, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_729, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_486, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_769, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_522, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_809, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_558, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_849, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(unsqueeze_594, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_889, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(unsqueeze_630, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(mul_929, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_666, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(mul_969, (8, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(unsqueeze_702, (1, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_1009, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(unsqueeze_738, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_1049, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_774, (1, 32, 1, 1), (32, 1, 1, 1))
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
    buf6 = empty((8, 1280, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mul_409.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(unsqueeze_198.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del buf4
    del convolution_80
    del mul_409
    del primals_102
    del squeeze_145
    del tangents_1
    del unsqueeze_198
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
    buf7 = aten.convolution_backward(buf6, add_248, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_248
    del buf6
    del primals_211
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((320, ), device='cpu', dtype=torch.float32)
    buf11 = empty((320, ), device='cpu', dtype=torch.float32)
    buf12 = empty((320, ), device='cpu', dtype=torch.float32)
    buf13 = buf8; del buf8  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_1(c_void_p(buf13.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(unsqueeze_210.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del buf11
    del convolution_79
    del primals_100
    del squeeze_142
    del unsqueeze_210
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf14 = aten.convolution_backward(buf13, mul_392, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_392
    del primals_210
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cpu', dtype=torch.float32)
    buf18 = reinterpret_tensor(buf17, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf17  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_2(c_void_p(buf18.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(add_243.data_ptr()), c_void_p(convolution_78.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf19 = aten.convolution_backward(buf18, mul_391, primals_208, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf18
    del mul_391
    del primals_208
    buf20 = buf19[0]
    buf21 = buf19[1]
    buf22 = buf19[2]
    del buf19
    buf23 = reinterpret_tensor(buf20, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf20  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_3(c_void_p(buf23.data_ptr()), c_void_p(convolution_77.data_ptr()))
    del convolution_77
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf24 = aten.convolution_backward(buf23, mean_15, primals_206, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf23
    del mean_15
    del primals_206
    buf25 = buf24[0]
    buf26 = buf24[1]
    buf27 = buf24[2]
    del buf24
    buf28 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf29 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf30 = buf15; del buf15  # reuse
    buf31 = buf29; del buf29  # reuse
    buf32 = buf30; del buf30  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_4(c_void_p(buf32.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(convolution_78.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(add_243.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(unsqueeze_222.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf28.data_ptr()))
    del add_243
    del convolution_76
    del convolution_78
    del primals_98
    del squeeze_139
    del unsqueeze_222
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf33 = aten.convolution_backward(buf32, mul_382, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf32
    del mul_382
    del primals_205
    buf34 = buf33[0]
    buf35 = buf33[1]
    del buf33
    buf36 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf37 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf38 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf39 = buf34; del buf34  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_5(c_void_p(buf39.data_ptr()), c_void_p(mul_449.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    del convolution_75
    del mul_449
    del primals_96
    del squeeze_136
    del unsqueeze_234
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf40 = aten.convolution_backward(buf39, add_233, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_233
    del buf39
    del primals_204
    buf41 = buf40[0]
    buf42 = buf40[1]
    del buf40
    buf43 = empty((192, ), device='cpu', dtype=torch.float32)
    buf44 = empty((192, ), device='cpu', dtype=torch.float32)
    buf45 = empty((192, ), device='cpu', dtype=torch.float32)
    buf46 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_6(c_void_p(buf41.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del convolution_74
    del primals_94
    del squeeze_133
    del unsqueeze_246
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf47 = aten.convolution_backward(buf46, mul_367, primals_203, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_367
    del primals_203
    buf48 = buf47[0]
    buf49 = buf47[1]
    del buf47
    buf50 = reinterpret_tensor(buf25, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf25  # reuse
    buf51 = reinterpret_tensor(buf50, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf50  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_7(c_void_p(buf51.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(add_227.data_ptr()), c_void_p(convolution_73.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf52 = aten.convolution_backward(buf51, mul_366, primals_201, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf51
    del mul_366
    del primals_201
    buf53 = buf52[0]
    buf54 = buf52[1]
    buf55 = buf52[2]
    del buf52
    buf56 = reinterpret_tensor(buf53, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf53  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_8(c_void_p(buf56.data_ptr()), c_void_p(convolution_72.data_ptr()))
    del convolution_72
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf57 = aten.convolution_backward(buf56, mean_14, primals_199, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf56
    del mean_14
    del primals_199
    buf58 = buf57[0]
    buf59 = buf57[1]
    buf60 = buf57[2]
    del buf57
    buf61 = buf37; del buf37  # reuse
    buf62 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf63 = buf48; del buf48  # reuse
    buf64 = buf62; del buf62  # reuse
    buf65 = buf63; del buf63  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_9(c_void_p(buf65.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(add_227.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf61.data_ptr()))
    del add_227
    del convolution_71
    del convolution_73
    del primals_92
    del squeeze_130
    del unsqueeze_258
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf66 = aten.convolution_backward(buf65, mul_357, primals_198, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf65
    del mul_357
    del primals_198
    buf67 = buf66[0]
    buf68 = buf66[1]
    del buf66
    buf69 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf70 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf71 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf72 = buf67; del buf67  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_10(c_void_p(buf72.data_ptr()), c_void_p(mul_489.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(unsqueeze_270.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    del convolution_70
    del mul_489
    del primals_90
    del squeeze_127
    del unsqueeze_270
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf73 = aten.convolution_backward(buf72, add_217, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_217
    del buf72
    del primals_197
    buf74 = buf73[0]
    buf75 = buf73[1]
    del buf73
    buf76 = buf44; del buf44  # reuse
    buf77 = empty((192, ), device='cpu', dtype=torch.float32)
    buf78 = empty((192, ), device='cpu', dtype=torch.float32)
    buf79 = buf46; del buf46  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_11(c_void_p(buf41.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del convolution_69
    del primals_88
    del squeeze_124
    del unsqueeze_282
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf80 = aten.convolution_backward(buf79, mul_342, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_342
    del primals_196
    buf81 = buf80[0]
    buf82 = buf80[1]
    del buf80
    buf83 = reinterpret_tensor(buf58, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf58  # reuse
    buf84 = reinterpret_tensor(buf83, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf83  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_12(c_void_p(buf84.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(add_211.data_ptr()), c_void_p(convolution_68.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf85 = aten.convolution_backward(buf84, mul_341, primals_194, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf84
    del mul_341
    del primals_194
    buf86 = buf85[0]
    buf87 = buf85[1]
    buf88 = buf85[2]
    del buf85
    buf89 = reinterpret_tensor(buf86, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf86  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_13(c_void_p(buf89.data_ptr()), c_void_p(convolution_67.data_ptr()))
    del convolution_67
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf90 = aten.convolution_backward(buf89, mean_13, primals_192, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf89
    del mean_13
    del primals_192
    buf91 = buf90[0]
    buf92 = buf90[1]
    buf93 = buf90[2]
    del buf90
    buf94 = buf70; del buf70  # reuse
    buf95 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf96 = buf81; del buf81  # reuse
    buf97 = buf95; del buf95  # reuse
    buf98 = buf96; del buf96  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_14(c_void_p(buf98.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(add_211.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(unsqueeze_294.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf94.data_ptr()))
    del add_211
    del convolution_66
    del convolution_68
    del primals_86
    del squeeze_121
    del unsqueeze_294
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf99 = aten.convolution_backward(buf98, mul_332, primals_191, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf98
    del mul_332
    del primals_191
    buf100 = buf99[0]
    buf101 = buf99[1]
    del buf99
    buf102 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf103 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf104 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf105 = buf100; del buf100  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_15(c_void_p(buf105.data_ptr()), c_void_p(mul_529.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    del convolution_65
    del mul_529
    del primals_84
    del squeeze_118
    del unsqueeze_306
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf106 = aten.convolution_backward(buf105, add_201, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_201
    del buf105
    del primals_190
    buf107 = buf106[0]
    buf108 = buf106[1]
    del buf106
    buf109 = buf77; del buf77  # reuse
    buf110 = empty((192, ), device='cpu', dtype=torch.float32)
    buf111 = buf79; del buf79  # reuse
    buf112 = buf110; del buf110  # reuse
    cpp_fused_add_native_batch_norm_backward_16(c_void_p(buf112.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_318.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()))
    del convolution_64
    del primals_82
    del squeeze_115
    del unsqueeze_318
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf113 = aten.convolution_backward(buf111, mul_317, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf111
    del mul_317
    del primals_189
    buf114 = buf113[0]
    buf115 = buf113[1]
    del buf113
    buf116 = reinterpret_tensor(buf91, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf91  # reuse
    buf117 = reinterpret_tensor(buf116, (8, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf116  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_17(c_void_p(buf117.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(add_195.data_ptr()), c_void_p(convolution_63.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf118 = aten.convolution_backward(buf117, mul_316, primals_187, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf117
    del mul_316
    del primals_187
    buf119 = buf118[0]
    buf120 = buf118[1]
    buf121 = buf118[2]
    del buf118
    buf122 = reinterpret_tensor(buf119, (8, 48, 1, 1), (48, 1, 1, 1), 0); del buf119  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_18(c_void_p(buf122.data_ptr()), c_void_p(convolution_62.data_ptr()))
    del convolution_62
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf123 = aten.convolution_backward(buf122, mean_12, primals_185, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf122
    del mean_12
    del primals_185
    buf124 = buf123[0]
    buf125 = buf123[1]
    buf126 = buf123[2]
    del buf123
    buf127 = buf103; del buf103  # reuse
    buf128 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf129 = buf114; del buf114  # reuse
    buf130 = buf128; del buf128  # reuse
    buf131 = buf129; del buf129  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_19(c_void_p(buf131.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(add_195.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf127.data_ptr()))
    del add_195
    del buf124
    del convolution_61
    del convolution_63
    del primals_80
    del squeeze_112
    del unsqueeze_330
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf132 = aten.convolution_backward(buf131, mul_307, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf131
    del mul_307
    del primals_184
    buf133 = buf132[0]
    buf134 = buf132[1]
    del buf132
    buf135 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf136 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf137 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf138 = buf133; del buf133  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_20(c_void_p(buf138.data_ptr()), c_void_p(mul_569.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()))
    del convolution_60
    del mul_569
    del primals_78
    del squeeze_109
    del unsqueeze_342
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf139 = aten.convolution_backward(buf138, add_185, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_185
    del buf138
    del primals_183
    buf140 = buf139[0]
    buf141 = buf139[1]
    del buf139
    buf142 = empty((192, ), device='cpu', dtype=torch.float32)
    buf143 = empty((192, ), device='cpu', dtype=torch.float32)
    buf144 = buf107; del buf107  # reuse
    buf146 = buf144; del buf144  # reuse
    buf145 = buf143; del buf143  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_21(c_void_p(buf146.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf142.data_ptr()))
    del buf140
    del buf41
    del buf74
    del convolution_59
    del primals_76
    del squeeze_106
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf147 = aten.convolution_backward(buf146, mul_292, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf146
    del mul_292
    del primals_182
    buf148 = buf147[0]
    buf149 = buf147[1]
    del buf147
    buf150 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf151 = reinterpret_tensor(buf150, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf150  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_22(c_void_p(buf151.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(add_180.data_ptr()), c_void_p(convolution_58.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf152 = aten.convolution_backward(buf151, mul_291, primals_180, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf151
    del mul_291
    del primals_180
    buf153 = buf152[0]
    buf154 = buf152[1]
    buf155 = buf152[2]
    del buf152
    buf156 = reinterpret_tensor(buf153, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf153  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_23(c_void_p(buf156.data_ptr()), c_void_p(convolution_57.data_ptr()))
    del convolution_57
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf157 = aten.convolution_backward(buf156, mean_11, primals_178, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf156
    del mean_11
    del primals_178
    buf158 = buf157[0]
    buf159 = buf157[1]
    buf160 = buf157[2]
    del buf157
    buf161 = empty((672, ), device='cpu', dtype=torch.float32)
    buf162 = empty((672, ), device='cpu', dtype=torch.float32)
    buf163 = buf148; del buf148  # reuse
    buf164 = buf162; del buf162  # reuse
    buf165 = buf163; del buf163  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_24(c_void_p(buf165.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(add_180.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf161.data_ptr()))
    del add_180
    del convolution_56
    del convolution_58
    del primals_74
    del squeeze_103
    del unsqueeze_366
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf166 = aten.convolution_backward(buf165, constant_pad_nd_4, primals_73, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf165
    del constant_pad_nd_4
    del primals_73
    buf167 = buf166[0]
    buf168 = buf166[1]
    del buf166
    buf169 = empty((672, ), device='cpu', dtype=torch.float32)
    buf170 = empty((672, ), device='cpu', dtype=torch.float32)
    buf171 = empty((672, ), device='cpu', dtype=torch.float32)
    buf172 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_25(c_void_p(buf167.data_ptr()), c_void_p(mul_609.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del buf167
    del convolution_55
    del mul_609
    del primals_71
    del squeeze_100
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf173 = aten.convolution_backward(buf172, add_170, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_170
    del buf172
    del primals_177
    buf174 = buf173[0]
    buf175 = buf173[1]
    del buf173
    buf176 = empty((112, ), device='cpu', dtype=torch.float32)
    buf177 = empty((112, ), device='cpu', dtype=torch.float32)
    buf178 = empty((112, ), device='cpu', dtype=torch.float32)
    buf179 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_26(c_void_p(buf174.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()))
    del convolution_54
    del primals_69
    del squeeze_97
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf180 = aten.convolution_backward(buf179, mul_267, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_267
    del primals_176
    buf181 = buf180[0]
    buf182 = buf180[1]
    del buf180
    buf183 = reinterpret_tensor(buf158, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf158  # reuse
    buf184 = reinterpret_tensor(buf183, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf183  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_27(c_void_p(buf184.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(add_164.data_ptr()), c_void_p(convolution_53.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf185 = aten.convolution_backward(buf184, mul_266, primals_174, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf184
    del mul_266
    del primals_174
    buf186 = buf185[0]
    buf187 = buf185[1]
    buf188 = buf185[2]
    del buf185
    buf189 = reinterpret_tensor(buf186, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf186  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_28(c_void_p(buf189.data_ptr()), c_void_p(convolution_52.data_ptr()))
    del convolution_52
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf190 = aten.convolution_backward(buf189, mean_10, primals_172, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf189
    del mean_10
    del primals_172
    buf191 = buf190[0]
    buf192 = buf190[1]
    buf193 = buf190[2]
    del buf190
    buf194 = buf170; del buf170  # reuse
    buf195 = empty((672, ), device='cpu', dtype=torch.float32)
    buf196 = buf181; del buf181  # reuse
    buf197 = buf195; del buf195  # reuse
    buf198 = buf196; del buf196  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_29(c_void_p(buf198.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(add_164.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf194.data_ptr()))
    del add_164
    del convolution_51
    del convolution_53
    del primals_67
    del squeeze_94
    del unsqueeze_402
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf199 = aten.convolution_backward(buf198, mul_257, primals_171, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf198
    del mul_257
    del primals_171
    buf200 = buf199[0]
    buf201 = buf199[1]
    del buf199
    buf202 = empty((672, ), device='cpu', dtype=torch.float32)
    buf203 = empty((672, ), device='cpu', dtype=torch.float32)
    buf204 = empty((672, ), device='cpu', dtype=torch.float32)
    buf205 = buf200; del buf200  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_30(c_void_p(buf205.data_ptr()), c_void_p(mul_649.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    del convolution_50
    del mul_649
    del primals_65
    del squeeze_91
    del unsqueeze_414
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf206 = aten.convolution_backward(buf205, add_154, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_154
    del buf205
    del primals_170
    buf207 = buf206[0]
    buf208 = buf206[1]
    del buf206
    buf209 = buf177; del buf177  # reuse
    buf210 = empty((112, ), device='cpu', dtype=torch.float32)
    buf211 = empty((112, ), device='cpu', dtype=torch.float32)
    buf212 = buf179; del buf179  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_31(c_void_p(buf174.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    del convolution_49
    del primals_63
    del squeeze_88
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf213 = aten.convolution_backward(buf212, mul_242, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf212
    del mul_242
    del primals_169
    buf214 = buf213[0]
    buf215 = buf213[1]
    del buf213
    buf216 = reinterpret_tensor(buf191, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf191  # reuse
    buf217 = reinterpret_tensor(buf216, (8, 672, 1, 1), (672, 1, 1, 1), 0); del buf216  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_32(c_void_p(buf217.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(add_148.data_ptr()), c_void_p(convolution_48.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf218 = aten.convolution_backward(buf217, mul_241, primals_167, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf217
    del mul_241
    del primals_167
    buf219 = buf218[0]
    buf220 = buf218[1]
    buf221 = buf218[2]
    del buf218
    buf222 = reinterpret_tensor(buf219, (8, 28, 1, 1), (28, 1, 1, 1), 0); del buf219  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_33(c_void_p(buf222.data_ptr()), c_void_p(convolution_47.data_ptr()))
    del convolution_47
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf223 = aten.convolution_backward(buf222, mean_9, primals_165, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf222
    del mean_9
    del primals_165
    buf224 = buf223[0]
    buf225 = buf223[1]
    buf226 = buf223[2]
    del buf223
    buf227 = buf203; del buf203  # reuse
    buf228 = empty((672, ), device='cpu', dtype=torch.float32)
    buf229 = buf214; del buf214  # reuse
    buf230 = buf228; del buf228  # reuse
    buf231 = buf229; del buf229  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_34(c_void_p(buf231.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(add_148.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf227.data_ptr()))
    del add_148
    del buf224
    del convolution_46
    del convolution_48
    del primals_61
    del squeeze_85
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf232 = aten.convolution_backward(buf231, mul_232, primals_164, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf231
    del mul_232
    del primals_164
    buf233 = buf232[0]
    buf234 = buf232[1]
    del buf232
    buf235 = empty((672, ), device='cpu', dtype=torch.float32)
    buf236 = empty((672, ), device='cpu', dtype=torch.float32)
    buf237 = empty((672, ), device='cpu', dtype=torch.float32)
    buf238 = buf233; del buf233  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_35(c_void_p(buf238.data_ptr()), c_void_p(mul_689.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    del buf236
    del convolution_45
    del mul_689
    del primals_59
    del squeeze_82
    del unsqueeze_450
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf239 = aten.convolution_backward(buf238, add_138, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_138
    del buf238
    del primals_163
    buf240 = buf239[0]
    buf241 = buf239[1]
    del buf239
    buf242 = buf210; del buf210  # reuse
    buf243 = empty((112, ), device='cpu', dtype=torch.float32)
    buf244 = buf174; del buf174  # reuse
    buf245 = buf243; del buf243  # reuse
    cpp_fused_add_native_batch_norm_backward_36(c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_462.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf242.data_ptr()))
    del buf207
    del buf240
    del convolution_44
    del primals_57
    del squeeze_79
    del unsqueeze_462
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf246 = aten.convolution_backward(buf244, mul_217, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf244
    del mul_217
    del primals_162
    buf247 = buf246[0]
    buf248 = buf246[1]
    del buf246
    buf249 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf250 = reinterpret_tensor(buf249, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf249  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_37(c_void_p(buf250.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(add_133.data_ptr()), c_void_p(convolution_43.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf251 = aten.convolution_backward(buf250, mul_216, primals_160, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf250
    del mul_216
    del primals_160
    buf252 = buf251[0]
    buf253 = buf251[1]
    buf254 = buf251[2]
    del buf251
    buf255 = reinterpret_tensor(buf252, (8, 20, 1, 1), (20, 1, 1, 1), 0); del buf252  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_38(c_void_p(buf255.data_ptr()), c_void_p(convolution_42.data_ptr()))
    del convolution_42
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf256 = aten.convolution_backward(buf255, mean_8, primals_158, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf255
    del mean_8
    del primals_158
    buf257 = buf256[0]
    buf258 = buf256[1]
    buf259 = buf256[2]
    del buf256
    buf260 = empty((480, ), device='cpu', dtype=torch.float32)
    buf261 = empty((480, ), device='cpu', dtype=torch.float32)
    buf262 = buf247; del buf247  # reuse
    buf263 = buf261; del buf261  # reuse
    buf264 = buf262; del buf262  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_39(c_void_p(buf264.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(add_133.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf260.data_ptr()))
    del add_133
    del convolution_41
    del convolution_43
    del primals_55
    del squeeze_76
    del unsqueeze_474
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf265 = aten.convolution_backward(buf264, mul_207, primals_157, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf264
    del mul_207
    del primals_157
    buf266 = buf265[0]
    buf267 = buf265[1]
    del buf265
    buf268 = empty((480, ), device='cpu', dtype=torch.float32)
    buf269 = empty((480, ), device='cpu', dtype=torch.float32)
    buf270 = empty((480, ), device='cpu', dtype=torch.float32)
    buf271 = buf266; del buf266  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_40(c_void_p(buf271.data_ptr()), c_void_p(mul_729.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_486.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    del convolution_40
    del mul_729
    del primals_53
    del squeeze_73
    del unsqueeze_486
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf272 = aten.convolution_backward(buf271, add_123, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_123
    del buf271
    del primals_156
    buf273 = buf272[0]
    buf274 = buf272[1]
    del buf272
    buf275 = empty((80, ), device='cpu', dtype=torch.float32)
    buf276 = empty((80, ), device='cpu', dtype=torch.float32)
    buf277 = empty((80, ), device='cpu', dtype=torch.float32)
    buf278 = reinterpret_tensor(buf13, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf13  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_41(c_void_p(buf273.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    del convolution_39
    del primals_51
    del squeeze_70
    del unsqueeze_498
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf279 = aten.convolution_backward(buf278, mul_192, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_192
    del primals_155
    buf280 = buf279[0]
    buf281 = buf279[1]
    del buf279
    buf282 = reinterpret_tensor(buf257, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf257  # reuse
    buf283 = reinterpret_tensor(buf282, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf282  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_42(c_void_p(buf283.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(add_117.data_ptr()), c_void_p(convolution_38.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf284 = aten.convolution_backward(buf283, mul_191, primals_153, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf283
    del mul_191
    del primals_153
    buf285 = buf284[0]
    buf286 = buf284[1]
    buf287 = buf284[2]
    del buf284
    buf288 = reinterpret_tensor(buf285, (8, 20, 1, 1), (20, 1, 1, 1), 0); del buf285  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_43(c_void_p(buf288.data_ptr()), c_void_p(convolution_37.data_ptr()))
    del convolution_37
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf289 = aten.convolution_backward(buf288, mean_7, primals_151, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf288
    del mean_7
    del primals_151
    buf290 = buf289[0]
    buf291 = buf289[1]
    buf292 = buf289[2]
    del buf289
    buf293 = buf269; del buf269  # reuse
    buf294 = empty((480, ), device='cpu', dtype=torch.float32)
    buf295 = buf280; del buf280  # reuse
    buf296 = buf294; del buf294  # reuse
    buf297 = buf295; del buf295  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_44(c_void_p(buf297.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(add_117.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_510.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf293.data_ptr()))
    del add_117
    del convolution_36
    del convolution_38
    del primals_49
    del squeeze_67
    del unsqueeze_510
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf298 = aten.convolution_backward(buf297, mul_182, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf297
    del mul_182
    del primals_150
    buf299 = buf298[0]
    buf300 = buf298[1]
    del buf298
    buf301 = empty((480, ), device='cpu', dtype=torch.float32)
    buf302 = empty((480, ), device='cpu', dtype=torch.float32)
    buf303 = empty((480, ), device='cpu', dtype=torch.float32)
    buf304 = buf299; del buf299  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_45(c_void_p(buf304.data_ptr()), c_void_p(mul_769.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_522.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()))
    del convolution_35
    del mul_769
    del primals_47
    del squeeze_64
    del unsqueeze_522
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf305 = aten.convolution_backward(buf304, add_107, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_107
    del buf304
    del primals_149
    buf306 = buf305[0]
    buf307 = buf305[1]
    del buf305
    buf308 = buf276; del buf276  # reuse
    buf309 = empty((80, ), device='cpu', dtype=torch.float32)
    buf310 = empty((80, ), device='cpu', dtype=torch.float32)
    buf311 = buf278; del buf278  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_46(c_void_p(buf273.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_534.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()))
    del convolution_34
    del primals_45
    del squeeze_61
    del unsqueeze_534
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf312 = aten.convolution_backward(buf311, mul_167, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf311
    del mul_167
    del primals_148
    buf313 = buf312[0]
    buf314 = buf312[1]
    del buf312
    buf315 = reinterpret_tensor(buf290, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf290  # reuse
    buf316 = reinterpret_tensor(buf315, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf315  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_47(c_void_p(buf316.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(add_101.data_ptr()), c_void_p(convolution_33.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf317 = aten.convolution_backward(buf316, mul_166, primals_146, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf316
    del mul_166
    del primals_146
    buf318 = buf317[0]
    buf319 = buf317[1]
    buf320 = buf317[2]
    del buf317
    buf321 = reinterpret_tensor(buf318, (8, 20, 1, 1), (20, 1, 1, 1), 0); del buf318  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_48(c_void_p(buf321.data_ptr()), c_void_p(convolution_32.data_ptr()))
    del convolution_32
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf322 = aten.convolution_backward(buf321, mean_6, primals_144, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf321
    del mean_6
    del primals_144
    buf323 = buf322[0]
    buf324 = buf322[1]
    buf325 = buf322[2]
    del buf322
    buf326 = buf302; del buf302  # reuse
    buf327 = empty((480, ), device='cpu', dtype=torch.float32)
    buf328 = buf313; del buf313  # reuse
    buf329 = buf327; del buf327  # reuse
    buf330 = buf328; del buf328  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_49(c_void_p(buf330.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(add_101.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_546.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf326.data_ptr()))
    del add_101
    del buf323
    del convolution_31
    del convolution_33
    del primals_43
    del squeeze_58
    del unsqueeze_546
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf331 = aten.convolution_backward(buf330, mul_157, primals_143, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf330
    del mul_157
    del primals_143
    buf332 = buf331[0]
    buf333 = buf331[1]
    del buf331
    buf334 = empty((480, ), device='cpu', dtype=torch.float32)
    buf335 = empty((480, ), device='cpu', dtype=torch.float32)
    buf336 = empty((480, ), device='cpu', dtype=torch.float32)
    buf337 = buf332; del buf332  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_50(c_void_p(buf337.data_ptr()), c_void_p(mul_809.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_558.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    del buf335
    del convolution_30
    del mul_809
    del primals_41
    del squeeze_55
    del unsqueeze_558
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf338 = aten.convolution_backward(buf337, add_91, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_91
    del buf337
    del primals_142
    buf339 = buf338[0]
    buf340 = buf338[1]
    del buf338
    buf341 = buf309; del buf309  # reuse
    buf342 = empty((80, ), device='cpu', dtype=torch.float32)
    buf343 = buf273; del buf273  # reuse
    buf344 = buf342; del buf342  # reuse
    cpp_fused_add_native_batch_norm_backward_51(c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_570.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf341.data_ptr()))
    del buf306
    del buf339
    del convolution_29
    del primals_39
    del squeeze_52
    del unsqueeze_570
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf345 = aten.convolution_backward(buf343, mul_142, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf343
    del mul_142
    del primals_141
    buf346 = buf345[0]
    buf347 = buf345[1]
    del buf345
    buf348 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf349 = reinterpret_tensor(buf348, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf348  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_52(c_void_p(buf349.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(add_86.data_ptr()), c_void_p(convolution_28.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf350 = aten.convolution_backward(buf349, mul_141, primals_139, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf349
    del mul_141
    del primals_139
    buf351 = buf350[0]
    buf352 = buf350[1]
    buf353 = buf350[2]
    del buf350
    buf354 = reinterpret_tensor(buf351, (8, 10, 1, 1), (10, 1, 1, 1), 0); del buf351  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_53(c_void_p(buf354.data_ptr()), c_void_p(convolution_27.data_ptr()))
    del convolution_27
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf355 = aten.convolution_backward(buf354, mean_5, primals_137, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf354
    del mean_5
    del primals_137
    buf356 = buf355[0]
    buf357 = buf355[1]
    buf358 = buf355[2]
    del buf355
    buf359 = empty((240, ), device='cpu', dtype=torch.float32)
    buf360 = empty((240, ), device='cpu', dtype=torch.float32)
    buf361 = buf346; del buf346  # reuse
    buf362 = buf360; del buf360  # reuse
    buf363 = buf361; del buf361  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_54(c_void_p(buf363.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(add_86.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_582.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf359.data_ptr()))
    del add_86
    del convolution_26
    del convolution_28
    del primals_37
    del squeeze_49
    del unsqueeze_582
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf364 = aten.convolution_backward(buf363, constant_pad_nd_3, primals_36, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf363
    del constant_pad_nd_3
    del primals_36
    buf365 = buf364[0]
    buf366 = buf364[1]
    del buf364
    buf367 = empty((240, ), device='cpu', dtype=torch.float32)
    buf368 = empty((240, ), device='cpu', dtype=torch.float32)
    buf369 = empty((240, ), device='cpu', dtype=torch.float32)
    buf370 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_55(c_void_p(buf365.data_ptr()), c_void_p(mul_849.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_594.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    del buf365
    del convolution_25
    del mul_849
    del primals_34
    del squeeze_46
    del unsqueeze_594
    # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf371 = aten.convolution_backward(buf370, add_76, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_76
    del buf370
    del primals_136
    buf372 = buf371[0]
    buf373 = buf371[1]
    del buf371
    buf374 = empty((40, ), device='cpu', dtype=torch.float32)
    buf375 = empty((40, ), device='cpu', dtype=torch.float32)
    buf376 = empty((40, ), device='cpu', dtype=torch.float32)
    buf377 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_56(c_void_p(buf372.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_606.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del convolution_24
    del primals_32
    del squeeze_43
    del unsqueeze_606
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf378 = aten.convolution_backward(buf377, mul_117, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf377
    del mul_117
    del primals_135
    buf379 = buf378[0]
    buf380 = buf378[1]
    del buf378
    buf381 = reinterpret_tensor(buf356, (8, 240, 1, 1), (240, 1, 1920, 1920), 0); del buf356  # reuse
    buf382 = reinterpret_tensor(buf381, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf381  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_57(c_void_p(buf382.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(add_70.data_ptr()), c_void_p(convolution_23.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf383 = aten.convolution_backward(buf382, mul_116, primals_133, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf382
    del mul_116
    del primals_133
    buf384 = buf383[0]
    buf385 = buf383[1]
    buf386 = buf383[2]
    del buf383
    buf387 = reinterpret_tensor(buf384, (8, 10, 1, 1), (10, 1, 1, 1), 0); del buf384  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_58(c_void_p(buf387.data_ptr()), c_void_p(convolution_22.data_ptr()))
    del convolution_22
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf388 = aten.convolution_backward(buf387, mean_4, primals_131, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf387
    del mean_4
    del primals_131
    buf389 = buf388[0]
    buf390 = buf388[1]
    buf391 = buf388[2]
    del buf388
    buf392 = buf368; del buf368  # reuse
    buf393 = empty((240, ), device='cpu', dtype=torch.float32)
    buf394 = buf379; del buf379  # reuse
    buf395 = buf393; del buf393  # reuse
    buf396 = buf394; del buf394  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_59(c_void_p(buf396.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(add_70.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_618.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf392.data_ptr()))
    del add_70
    del buf389
    del convolution_21
    del convolution_23
    del primals_30
    del squeeze_40
    del unsqueeze_618
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf397 = aten.convolution_backward(buf396, mul_107, primals_130, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf396
    del mul_107
    del primals_130
    buf398 = buf397[0]
    buf399 = buf397[1]
    del buf397
    buf400 = empty((240, ), device='cpu', dtype=torch.float32)
    buf401 = empty((240, ), device='cpu', dtype=torch.float32)
    buf402 = empty((240, ), device='cpu', dtype=torch.float32)
    buf403 = buf398; del buf398  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_60(c_void_p(buf403.data_ptr()), c_void_p(mul_889.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_630.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()))
    del buf401
    del convolution_20
    del mul_889
    del primals_28
    del squeeze_37
    del unsqueeze_630
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf404 = aten.convolution_backward(buf403, add_60, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_60
    del buf403
    del primals_129
    buf405 = buf404[0]
    buf406 = buf404[1]
    del buf404
    buf407 = buf375; del buf375  # reuse
    buf408 = empty((40, ), device='cpu', dtype=torch.float32)
    buf409 = empty((40, ), device='cpu', dtype=torch.float32)
    buf410 = buf372; del buf372  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_61(c_void_p(buf410.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_642.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    del buf405
    del buf408
    del convolution_19
    del primals_26
    del squeeze_34
    del unsqueeze_642
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf411 = aten.convolution_backward(buf410, mul_92, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf410
    del mul_92
    del primals_128
    buf412 = buf411[0]
    buf413 = buf411[1]
    del buf411
    buf414 = reinterpret_tensor(buf136, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf136  # reuse
    buf415 = reinterpret_tensor(buf414, (8, 144, 1, 1), (144, 1, 1, 1), 0); del buf414  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_62(c_void_p(buf415.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(add_55.data_ptr()), c_void_p(convolution_18.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf416 = aten.convolution_backward(buf415, mul_91, primals_126, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf415
    del mul_91
    del primals_126
    buf417 = buf416[0]
    buf418 = buf416[1]
    buf419 = buf416[2]
    del buf416
    buf420 = reinterpret_tensor(buf417, (8, 6, 1, 1), (6, 1, 1, 1), 0); del buf417  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_63(c_void_p(buf420.data_ptr()), c_void_p(convolution_17.data_ptr()))
    del convolution_17
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf421 = aten.convolution_backward(buf420, mean_3, primals_124, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf420
    del mean_3
    del primals_124
    buf422 = buf421[0]
    buf423 = buf421[1]
    buf424 = buf421[2]
    del buf421
    buf425 = empty((144, ), device='cpu', dtype=torch.float32)
    buf426 = empty((144, ), device='cpu', dtype=torch.float32)
    buf427 = buf412; del buf412  # reuse
    buf428 = buf426; del buf426  # reuse
    buf429 = buf427; del buf427  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64(c_void_p(buf429.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(add_55.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_654.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf425.data_ptr()))
    del add_55
    del convolution_16
    del convolution_18
    del primals_24
    del squeeze_31
    del unsqueeze_654
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf430 = aten.convolution_backward(buf429, constant_pad_nd_2, primals_23, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 144, [True, True, False])
    del buf429
    del constant_pad_nd_2
    del primals_23
    buf431 = buf430[0]
    buf432 = buf430[1]
    del buf430
    buf433 = empty((144, ), device='cpu', dtype=torch.float32)
    buf434 = empty((144, ), device='cpu', dtype=torch.float32)
    buf435 = empty((144, ), device='cpu', dtype=torch.float32)
    buf436 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_65(c_void_p(buf431.data_ptr()), c_void_p(mul_929.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_666.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    del buf431
    del convolution_15
    del mul_929
    del primals_21
    del squeeze_28
    del unsqueeze_666
    # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf437 = aten.convolution_backward(buf436, add_45, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_45
    del buf436
    del primals_123
    buf438 = buf437[0]
    buf439 = buf437[1]
    del buf437
    buf440 = empty((24, ), device='cpu', dtype=torch.float32)
    buf441 = empty((24, ), device='cpu', dtype=torch.float32)
    buf442 = empty((24, ), device='cpu', dtype=torch.float32)
    buf443 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_66(c_void_p(buf438.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_678.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()))
    del convolution_14
    del primals_19
    del squeeze_25
    del unsqueeze_678
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf444 = aten.convolution_backward(buf443, mul_67, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf443
    del mul_67
    del primals_122
    buf445 = buf444[0]
    buf446 = buf444[1]
    del buf444
    buf447 = reinterpret_tensor(buf422, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf422  # reuse
    buf448 = reinterpret_tensor(buf447, (8, 144, 1, 1), (144, 1, 1, 1), 0); del buf447  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_67(c_void_p(buf448.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(add_39.data_ptr()), c_void_p(convolution_13.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf449 = aten.convolution_backward(buf448, mul_66, primals_120, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf448
    del mul_66
    del primals_120
    buf450 = buf449[0]
    buf451 = buf449[1]
    buf452 = buf449[2]
    del buf449
    buf453 = reinterpret_tensor(buf450, (8, 6, 1, 1), (6, 1, 1, 1), 0); del buf450  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_68(c_void_p(buf453.data_ptr()), c_void_p(convolution_12.data_ptr()))
    del convolution_12
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf454 = aten.convolution_backward(buf453, mean_2, primals_118, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf453
    del mean_2
    del primals_118
    buf455 = buf454[0]
    buf456 = buf454[1]
    buf457 = buf454[2]
    del buf454
    buf458 = buf434; del buf434  # reuse
    buf459 = empty((144, ), device='cpu', dtype=torch.float32)
    buf460 = buf445; del buf445  # reuse
    buf461 = buf459; del buf459  # reuse
    buf462 = buf460; del buf460  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_69(c_void_p(buf462.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(add_39.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_690.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf458.data_ptr()))
    del add_39
    del buf455
    del convolution_11
    del convolution_13
    del primals_17
    del squeeze_22
    del unsqueeze_690
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf463 = aten.convolution_backward(buf462, mul_57, primals_117, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
    del buf462
    del mul_57
    del primals_117
    buf464 = buf463[0]
    buf465 = buf463[1]
    del buf463
    buf466 = empty((144, ), device='cpu', dtype=torch.float32)
    buf467 = empty((144, ), device='cpu', dtype=torch.float32)
    buf468 = empty((144, ), device='cpu', dtype=torch.float32)
    buf469 = buf464; del buf464  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_70(c_void_p(buf469.data_ptr()), c_void_p(mul_969.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_702.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()))
    del buf467
    del convolution_10
    del mul_969
    del primals_15
    del squeeze_19
    del unsqueeze_702
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf470 = aten.convolution_backward(buf469, add_29, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_29
    del buf469
    del primals_116
    buf471 = buf470[0]
    buf472 = buf470[1]
    del buf470
    buf473 = buf441; del buf441  # reuse
    buf474 = empty((24, ), device='cpu', dtype=torch.float32)
    buf475 = empty((24, ), device='cpu', dtype=torch.float32)
    buf476 = buf438; del buf438  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_71(c_void_p(buf476.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_714.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()))
    del buf471
    del buf474
    del convolution_9
    del primals_13
    del squeeze_16
    del unsqueeze_714
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf477 = aten.convolution_backward(buf476, mul_42, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf476
    del mul_42
    del primals_115
    buf478 = buf477[0]
    buf479 = buf477[1]
    del buf477
    buf480 = empty_strided((8, 96, 1, 1), (96, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf481 = reinterpret_tensor(buf480, (8, 96, 1, 1), (96, 1, 1, 1), 0); del buf480  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_72(c_void_p(buf481.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_8.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf482 = aten.convolution_backward(buf481, mul_41, primals_113, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf481
    del mul_41
    del primals_113
    buf483 = buf482[0]
    buf484 = buf482[1]
    buf485 = buf482[2]
    del buf482
    buf486 = reinterpret_tensor(buf483, (8, 4, 1, 1), (4, 1, 1, 1), 0); del buf483  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_73(c_void_p(buf486.data_ptr()), c_void_p(convolution_7.data_ptr()))
    del convolution_7
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf487 = aten.convolution_backward(buf486, mean_1, primals_111, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_1
    del primals_111
    buf488 = buf487[0]
    buf489 = buf487[1]
    buf490 = buf487[2]
    del buf487
    buf491 = empty((96, ), device='cpu', dtype=torch.float32)
    buf492 = empty((96, ), device='cpu', dtype=torch.float32)
    buf493 = buf478; del buf478  # reuse
    buf494 = buf492; del buf492  # reuse
    buf495 = buf493; del buf493  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_74(c_void_p(buf495.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_726.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf491.data_ptr()))
    del add_24
    del buf488
    del convolution_6
    del convolution_8
    del primals_11
    del squeeze_13
    del unsqueeze_726
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf496 = aten.convolution_backward(buf495, constant_pad_nd_1, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 96, [True, True, False])
    del buf495
    del constant_pad_nd_1
    del primals_10
    buf497 = buf496[0]
    buf498 = buf496[1]
    del buf496
    buf499 = empty((96, ), device='cpu', dtype=torch.float32)
    buf500 = empty((96, ), device='cpu', dtype=torch.float32)
    buf501 = empty((96, ), device='cpu', dtype=torch.float32)
    buf502 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_backward_mul_native_batch_norm_backward_75(c_void_p(buf497.data_ptr()), c_void_p(mul_1009.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_738.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()))
    del buf497
    del buf500
    del convolution_5
    del mul_1009
    del primals_8
    del squeeze_10
    del unsqueeze_738
    # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf503 = aten.convolution_backward(buf502, add_14, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_14
    del buf502
    del primals_110
    buf504 = buf503[0]
    buf505 = buf503[1]
    del buf503
    buf506 = empty((16, ), device='cpu', dtype=torch.float32)
    buf507 = empty((16, ), device='cpu', dtype=torch.float32)
    buf508 = empty((16, ), device='cpu', dtype=torch.float32)
    buf509 = buf504; del buf504  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_76(c_void_p(buf509.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_750.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()))
    del buf507
    del convolution_4
    del primals_6
    del squeeze_7
    del unsqueeze_750
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf510 = aten.convolution_backward(buf509, mul_17, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf509
    del mul_17
    del primals_109
    buf511 = buf510[0]
    buf512 = buf510[1]
    del buf510
    buf513 = empty_strided((8, 32, 1, 1), (32, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf514 = reinterpret_tensor(buf513, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf513  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_77(c_void_p(buf514.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(add_9.data_ptr()), c_void_p(convolution_3.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf515 = aten.convolution_backward(buf514, mul_16, primals_107, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf514
    del mul_16
    del primals_107
    buf516 = buf515[0]
    buf517 = buf515[1]
    buf518 = buf515[2]
    del buf515
    buf519 = reinterpret_tensor(buf516, (8, 8, 1, 1), (8, 1, 1, 1), 0); del buf516  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_78(c_void_p(buf519.data_ptr()), c_void_p(convolution_2.data_ptr()))
    del convolution_2
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf520 = aten.convolution_backward(buf519, mean, primals_105, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf519
    del mean
    del primals_105
    buf521 = buf520[0]
    buf522 = buf520[1]
    buf523 = buf520[2]
    del buf520
    buf524 = reinterpret_tensor(buf486, (32, ), (1, ), 0); del buf486  # reuse
    buf525 = empty((32, ), device='cpu', dtype=torch.float32)
    buf526 = buf511; del buf511  # reuse
    buf527 = buf525; del buf525  # reuse
    buf528 = buf526; del buf526  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_79(c_void_p(buf528.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(add_9.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_762.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf524.data_ptr()))
    del add_9
    del buf521
    del convolution_1
    del convolution_3
    del primals_4
    del squeeze_4
    del unsqueeze_762
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf529 = aten.convolution_backward(buf528, mul_7, primals_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf528
    del mul_7
    del primals_104
    buf530 = buf529[0]
    buf531 = buf529[1]
    del buf529
    buf532 = empty((32, ), device='cpu', dtype=torch.float32)
    buf533 = empty((32, ), device='cpu', dtype=torch.float32)
    buf534 = empty((32, ), device='cpu', dtype=torch.float32)
    buf535 = buf530; del buf530  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_80(c_void_p(buf535.data_ptr()), c_void_p(mul_1049.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_774.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()))
    del buf533
    del convolution
    del mul_1049
    del primals_2
    del squeeze_1
    del unsqueeze_774
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf536 = aten.convolution_backward(buf535, constant_pad_nd, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf535
    del constant_pad_nd
    del primals_1
    buf537 = buf536[1]
    return (buf537, buf534, buf532, buf527, buf524, buf508, buf506, buf501, buf499, buf498, buf494, buf491, buf475, buf473, buf468, buf466, buf461, buf458, buf442, buf440, buf435, buf433, buf432, buf428, buf425, buf409, buf407, buf402, buf400, buf395, buf392, buf376, buf374, buf369, buf367, buf366, buf362, buf359, buf344, buf341, buf336, buf334, buf329, buf326, buf310, buf308, buf303, buf301, buf296, buf293, buf277, buf275, buf270, buf268, buf263, buf260, buf245, buf242, buf237, buf235, buf230, buf227, buf211, buf209, buf204, buf202, buf197, buf194, buf178, buf176, buf171, buf169, buf168, buf164, buf161, buf145, buf142, buf137, buf135, buf130, buf127, buf112, buf109, buf104, buf102, buf97, buf94, buf78, buf76, buf71, buf69, buf64, buf61, buf45, buf43, buf38, buf36, buf31, buf28, buf12, buf10, buf5, buf3, buf531, buf522, buf523, buf517, buf518, buf512, buf505, buf489, buf490, buf484, buf485, buf479, buf472, buf465, buf456, buf457, buf451, buf452, buf446, buf439, buf423, buf424, buf418, buf419, buf413, buf406, buf399, buf390, buf391, buf385, buf386, buf380, buf373, buf357, buf358, buf352, buf353, buf347, buf340, buf333, buf324, buf325, buf319, buf320, buf314, buf307, buf300, buf291, buf292, buf286, buf287, buf281, buf274, buf267, buf258, buf259, buf253, buf254, buf248, buf241, buf234, buf225, buf226, buf220, buf221, buf215, buf208, buf201, buf192, buf193, buf187, buf188, buf182, buf175, buf159, buf160, buf154, buf155, buf149, buf141, buf134, buf125, buf126, buf120, buf121, buf115, buf108, buf101, buf92, buf93, buf87, buf88, buf82, buf75, buf68, buf59, buf60, buf54, buf55, buf49, buf42, buf35, buf26, buf27, buf21, buf22, buf16, buf9, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    constant_pad_nd = rand_strided((8, 3, 225, 225), (151875, 1, 675, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    mul_7 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    add_9 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    mul_17 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    add_14 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    constant_pad_nd_1 = rand_strided((8, 96, 113, 113), (1225824, 1, 10848, 96), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    add_24 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 4, 1, 1), (4, 1, 4, 4), device='cpu', dtype=torch.float32)
    mul_41 = rand_strided((8, 4, 1, 1), (4, 1, 4, 4), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    mul_42 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_29 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    mul_57 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    add_39 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    mul_66 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    mul_67 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_45 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    constant_pad_nd_2 = rand_strided((8, 144, 59, 59), (501264, 1, 8496, 144), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    add_55 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    mul_91 = rand_strided((8, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    mul_92 = rand_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_60 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    mul_107 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    add_70 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    mul_116 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    mul_117 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_76 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    constant_pad_nd_3 = rand_strided((8, 240, 29, 29), (201840, 1, 6960, 240), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    add_86 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    mul_141 = rand_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    mul_142 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_91 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    mul_157 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_101 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_166 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_167 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_107 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    mul_182 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_117 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_191 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_192 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_123 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    mul_207 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    add_133 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_216 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_217 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_138 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    mul_232 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    add_148 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_241 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_242 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_154 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    mul_257 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    add_164 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_266 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_267 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_170 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    constant_pad_nd_4 = rand_strided((8, 672, 17, 17), (194208, 1, 11424, 672), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    add_180 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_291 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_292 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_185 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    mul_307 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    add_195 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mean_12 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_316 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_317 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_201 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    mul_332 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    add_211 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mean_13 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_341 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_342 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_217 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    mul_357 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    add_227 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mean_14 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_366 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_367 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_233 = rand_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    mul_382 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    add_243 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mean_15 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_391 = rand_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_78 = rand_strided((8, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_392 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    add_248 = rand_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    view = rand_strided((8, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    mul_409 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_449 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_489 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_529 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_569 = rand_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 1152, 1, 1), (1152, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_609 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_649 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_689 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_729 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_769 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_809 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_849 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_889 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_929 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_969 = rand_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1009 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1049 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_120, primals_122, primals_123, primals_124, primals_126, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, constant_pad_nd, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, add_9, mean, convolution_2, mul_16, convolution_3, mul_17, convolution_4, squeeze_7, add_14, convolution_5, squeeze_10, constant_pad_nd_1, convolution_6, squeeze_13, add_24, mean_1, convolution_7, mul_41, convolution_8, mul_42, convolution_9, squeeze_16, add_29, convolution_10, squeeze_19, mul_57, convolution_11, squeeze_22, add_39, mean_2, convolution_12, mul_66, convolution_13, mul_67, convolution_14, squeeze_25, add_45, convolution_15, squeeze_28, constant_pad_nd_2, convolution_16, squeeze_31, add_55, mean_3, convolution_17, mul_91, convolution_18, mul_92, convolution_19, squeeze_34, add_60, convolution_20, squeeze_37, mul_107, convolution_21, squeeze_40, add_70, mean_4, convolution_22, mul_116, convolution_23, mul_117, convolution_24, squeeze_43, add_76, convolution_25, squeeze_46, constant_pad_nd_3, convolution_26, squeeze_49, add_86, mean_5, convolution_27, mul_141, convolution_28, mul_142, convolution_29, squeeze_52, add_91, convolution_30, squeeze_55, mul_157, convolution_31, squeeze_58, add_101, mean_6, convolution_32, mul_166, convolution_33, mul_167, convolution_34, squeeze_61, add_107, convolution_35, squeeze_64, mul_182, convolution_36, squeeze_67, add_117, mean_7, convolution_37, mul_191, convolution_38, mul_192, convolution_39, squeeze_70, add_123, convolution_40, squeeze_73, mul_207, convolution_41, squeeze_76, add_133, mean_8, convolution_42, mul_216, convolution_43, mul_217, convolution_44, squeeze_79, add_138, convolution_45, squeeze_82, mul_232, convolution_46, squeeze_85, add_148, mean_9, convolution_47, mul_241, convolution_48, mul_242, convolution_49, squeeze_88, add_154, convolution_50, squeeze_91, mul_257, convolution_51, squeeze_94, add_164, mean_10, convolution_52, mul_266, convolution_53, mul_267, convolution_54, squeeze_97, add_170, convolution_55, squeeze_100, constant_pad_nd_4, convolution_56, squeeze_103, add_180, mean_11, convolution_57, mul_291, convolution_58, mul_292, convolution_59, squeeze_106, add_185, convolution_60, squeeze_109, mul_307, convolution_61, squeeze_112, add_195, mean_12, convolution_62, mul_316, convolution_63, mul_317, convolution_64, squeeze_115, add_201, convolution_65, squeeze_118, mul_332, convolution_66, squeeze_121, add_211, mean_13, convolution_67, mul_341, convolution_68, mul_342, convolution_69, squeeze_124, add_217, convolution_70, squeeze_127, mul_357, convolution_71, squeeze_130, add_227, mean_14, convolution_72, mul_366, convolution_73, mul_367, convolution_74, squeeze_133, add_233, convolution_75, squeeze_136, mul_382, convolution_76, squeeze_139, add_243, mean_15, convolution_77, mul_391, convolution_78, mul_392, convolution_79, squeeze_142, add_248, convolution_80, squeeze_145, view, permute_1, mul_409, unsqueeze_198, unsqueeze_210, unsqueeze_222, mul_449, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_489, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_529, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_569, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_609, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_649, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_689, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_729, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_769, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_809, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_849, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_889, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_929, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_969, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1009, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1049, unsqueeze_774, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_efficientnet_b0', benchmark_compiled_module)
